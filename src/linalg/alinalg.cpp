#include "alinalg.hpp"
#include <iostream>
#include <vector>
#include <cstring>
#include <limits>

namespace fvens {

StatusCode createSystemVector(const UMesh<freal,NDIM> *const m, const int nvars, Vec *const v)
{
	StatusCode ierr = VecCreateMPI(PETSC_COMM_WORLD, m->gnelem()*nvars, m->gnelemglobal()*nvars, v);
	CHKERRQ(ierr);
	ierr = VecSetFromOptions(*v); CHKERRQ(ierr);
	return ierr;
}

StatusCode createGhostedSystemVector(const UMesh<freal,NDIM> *const m, const int nvars, Vec *const v)
{
	StatusCode ierr = 0;

	const std::vector<fint> globindices = m->getConnectivityGlobalIndices();

	ierr = VecCreateGhostBlock(PETSC_COMM_WORLD, nvars, m->gnelem()*nvars,
	                           m->gnelemglobal()*nvars, m->gnConnFace(),
	                           globindices.data(), v);
	CHKERRQ(ierr);
	ierr = VecSetFromOptions(*v); CHKERRQ(ierr);
	return ierr;
}

template <int nvars>
static StatusCode setJacobianSizes(const UMesh<freal,NDIM> *const m, Mat A) 
{
	StatusCode ierr = 0;
	ierr = MatSetSizes(A, m->gnelem()*nvars, m->gnelem()*nvars,
	                   m->gnelemglobal()*nvars, m->gnelemglobal()*nvars);
	CHKERRQ(ierr);
	ierr = MatSetBlockSize(A, nvars); CHKERRQ(ierr);
	return ierr;
}

template <int nvars>
StatusCode setJacobianPreallocation(const UMesh<freal,NDIM> *const m, Mat A) 
{
	StatusCode ierr = 0;

	// set block preallocation
	{
		std::vector<PetscInt> dnnz(m->gnelem());
		for(fint iel = 0; iel < m->gnelem(); iel++)
		{
			dnnz[iel] = m->gnfael(iel)+1;
		}

		std::vector<PetscInt> onnz(m->gnelem(),0);
		for(fint iface = 0; iface < m->gnConnFace(); iface++)
			onnz[m->gconnface(iface,0)] = 1;

		ierr = MatSeqBAIJSetPreallocation(A, nvars, 0, &dnnz[0]); CHKERRQ(ierr);
		ierr = MatMPIBAIJSetPreallocation(A, nvars, 0, &dnnz[0], 0, &onnz[0]); CHKERRQ(ierr);
	}

	// set scalar (non-block) preallocation
	{
		std::vector<PetscInt> dnnz(m->gnelem()*nvars);
		for(fint iel = 0; iel < m->gnelem(); iel++)
		{
			for(int i = 0; i < nvars; i++) {
				dnnz[iel*nvars+i] = (m->gnfael(iel)+1)*nvars;
			}
		}

		std::vector<PetscInt> onnz(m->gnelem()*nvars,0);
		for(fint iface = 0; iface < m->gnConnFace(); iface++)
		{
			for(int i = 0; i < nvars; i++)
				onnz[m->gconnface(iface,0)*nvars+i] = nvars;
		}

		ierr = MatSeqAIJSetPreallocation(A, 0, &dnnz[0]); CHKERRQ(ierr);
		ierr = MatMPIAIJSetPreallocation(A, 0, &dnnz[0], 0, &onnz[0]); CHKERRQ(ierr);
	}

	return ierr;
}

template StatusCode setJacobianPreallocation<1>(const UMesh<freal,NDIM> *const m, Mat A);
template StatusCode setJacobianPreallocation<NVARS>(const UMesh<freal,NDIM> *const m, Mat A);

template <int nvars>
StatusCode setupSystemMatrix(const UMesh<freal,NDIM> *const m, Mat *const A)
{
	StatusCode ierr = 0;
	ierr = MatCreate(PETSC_COMM_WORLD, A); CHKERRQ(ierr);
	ierr = MatSetType(*A, MATMPIBAIJ); CHKERRQ(ierr);

	ierr = setJacobianSizes<nvars>(m, *A); CHKERRQ(ierr);

	ierr = MatSetFromOptions(*A); CHKERRQ(ierr);

	ierr = setJacobianPreallocation<nvars>(m, *A); CHKERRQ(ierr);

	ierr = MatSetUp(*A); CHKERRQ(ierr);

	// MatType mtype;
	// ierr = MatGetType(*A, &mtype); CHKERRQ(ierr);
	// if(!strcmp(mtype, MATMPIBAIJ) || !strcmp(mtype,MATSEQBAIJ) || !strcmp(mtype,MATSEQAIJ)
	//    || !strcmp(mtype,MATMPIBAIJMKL) || !strcmp(mtype,MATSEQBAIJMKL) || !strcmp(mtype,MATSEQAIJMKL))
	// {
	// 	// This is only available for MATMPIBAIJ, it seems, but we know it also works for seq aij
	// 	//  and seq baij. Hopefully it also works for MKL Mats.
	// 	//  But it complains in case of mpi aij.
	// 	ierr = MatSetOption(*A, MAT_USE_HASH_TABLE, PETSC_TRUE); CHKERRQ(ierr);
	// }

	ierr = MatSetOption(*A, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE); CHKERRQ(ierr);

	return ierr;
}

template StatusCode setupSystemMatrix<NVARS>(const UMesh<freal,NDIM> *const m, Mat *const A);
template StatusCode setupSystemMatrix<1>(const UMesh<freal,NDIM> *const m, Mat *const A);

template<int nvars>
MatrixFreeSpatialJacobian<nvars>::MatrixFreeSpatialJacobian(const Spatial<freal,nvars> *const s)
	: spatial{s}, eps{1e-7}
{
	PetscBool set = PETSC_FALSE;
	PetscOptionsGetReal(NULL, NULL, "-matrix_free_difference_step", &eps, &set);
}

template<int nvars>
int MatrixFreeSpatialJacobian<nvars>::set_state(const Vec u_state, const Vec r_state,
		const Vec dtms) 
{
	u = u_state;
	res = r_state;
	mdt = dtms;
	return 0;
}

template<int nvars>
StatusCode MatrixFreeSpatialJacobian<nvars>::apply(const Vec x, Vec y) const
{
	StatusCode ierr = 0;
	Vec dummy = NULL;

	if(!spatial)
		SETERRQ(PETSC_COMM_SELF, PETSC_ERR_POINTER,
		        "Spatial context not set!");

	const UMesh<freal,NDIM> *const m = spatial->mesh();
	//ierr = VecSet(y, 0.0); CHKERRQ(ierr);

	Vec aux, yg;
	ierr = VecDuplicate(u, &aux); CHKERRQ(ierr);
	ierr = VecDuplicate(res, &yg); CHKERRQ(ierr);

	PetscScalar xnorm = 0;
	ierr = VecNorm(x, NORM_2, &xnorm); CHKERRQ(ierr);

#ifdef DEBUG
	if(xnorm < 10.0*std::numeric_limits<freal>::epsilon())
		SETERRQ(PETSC_COMM_SELF, PETSC_ERR_FP,
				"Norm of offset is too small for finite difference Jacobian!");
#endif
	const freal pertmag = eps/xnorm;

	// aux <- u + eps/xnorm * x ;    y <- 0
	{
		ConstVecHandler<PetscScalar> uh(u);
		const PetscScalar *const ur = uh.getArray();
		ConstVecHandler<PetscScalar> xh(x);
		const PetscScalar *const xr = xh.getArray();
		MutableVecHandler<PetscScalar> auxh(aux);
		PetscScalar *const auxr = auxh.getArray(); 
		MutableVecHandler<PetscScalar> ygh(yg);
		PetscScalar *const ygr = ygh.getArray(); 

#pragma omp parallel for simd default(shared)
		for(fint i = 0; i < m->gnelem()*nvars; i++) {
			ygr[i] = 0;
			auxr[i] = ur[i] + pertmag * xr[i];
		}

#pragma omp parallel for simd default(shared)
		for(fint i = m->gnelem(); i < m->gnelem()+m->gnConnFace(); i++) {
			ygr[i] = 0;
		}
	}

	ierr = VecGhostUpdateBegin(aux, INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);
	ierr = VecGhostUpdateEnd(aux, INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);

	// y <- -r(u + eps/xnorm * x)
	ierr = spatial->compute_residual(aux, yg, false, dummy); CHKERRQ(ierr);

	ierr = VecGhostUpdateBegin(yg, ADD_VALUES, SCATTER_REVERSE); CHKERRQ(ierr);
	ierr = VecGhostUpdateEnd(yg, ADD_VALUES, SCATTER_REVERSE); CHKERRQ(ierr);

	// y <- vol/dt x + (-(-r(u + eps/xnorm * x)) + (-r(u))) / eps |x|
	//    = vol/dt x + (r(u + eps/xnorm * x) - r(u)) / eps |x|
	/* We need to divide the difference by the step length scaled by the norm of x.
	 * We do NOT divide by epsilon, because we want the product of the Jacobian and x, which is
	 * the directional derivative (in the direction of x) multiplied by the norm of x.
	 */
	{
		ConstVecHandler<PetscScalar> xh(x);
		const PetscScalar *const xr = xh.getArray();
		ConstVecHandler<PetscScalar> resh(res);
		const PetscScalar *const resr = resh.getArray();
		ConstVecHandler<PetscScalar> mdth(mdt);
		const PetscScalar *const dtmr = mdth.getArray();
		ConstVecHandler<PetscScalar> ygh(yg);
		const PetscScalar *const ygr = ygh.getArray();
		MutableVecHandler<PetscScalar> yh(y);
		PetscScalar *const yr = yh.getArray(); 

#pragma omp parallel for simd default(shared)
		for(fint iel = 0; iel < m->gnelem(); iel++)
		{
			for(int i = 0; i < nvars; i++) {
				// finally, add the pseudo-time term (Vol/dt du = Vol/dt x)
				yr[iel*nvars+i] = dtmr[iel]*xr[iel*nvars+i]
					+ (-ygr[iel*nvars+i] + resr[iel*nvars+i])/pertmag;
			}
		}
	}
	
	ierr = VecDestroy(&aux); CHKERRQ(ierr);
	ierr = VecDestroy(&yg); CHKERRQ(ierr);
	return ierr;
}

template class MatrixFreeSpatialJacobian<NVARS>;
template class MatrixFreeSpatialJacobian<1>;

/// The function called by PETSc to carry out a Jacobian-vector product
template <int nvars>
StatusCode matrixfree_apply(Mat A, Vec x, Vec y)
{
	StatusCode ierr = 0;
	MatrixFreeSpatialJacobian<nvars> *mfmat;
	ierr = MatShellGetContext(A, (void*)&mfmat); CHKERRQ(ierr);
	ierr = mfmat->apply(x,y); CHKERRQ(ierr);
	return ierr;
}

/// Function called by PETSc to cleanup the matrix-free mat
template <int nvars>
StatusCode matrixfree_destroy(Mat A)
{
	StatusCode ierr = 0;
	MatrixFreeSpatialJacobian<nvars> *mfmat;
	ierr = MatShellGetContext(A, (void*)&mfmat); CHKERRQ(ierr);
	delete mfmat;
	return ierr;
}

template <int nvars>
StatusCode create_matrixfree_jacobian(const Spatial<freal,nvars> *const s, Mat *const A)
{
	StatusCode ierr = 0;

	const UMesh<freal,NDIM> *const m = s->mesh();
	MatrixFreeSpatialJacobian<nvars> *const mfj = new MatrixFreeSpatialJacobian<nvars>(s);
	
	ierr = MatCreate(PETSC_COMM_WORLD, A); CHKERRQ(ierr);
	ierr = setJacobianSizes<nvars>(m, *A); CHKERRQ(ierr);
	ierr = MatSetType(*A, MATSHELL); CHKERRQ(ierr);

	ierr = MatShellSetContext(*A, (void*)mfj); CHKERRQ(ierr);
	ierr = MatShellSetOperation(*A, MATOP_MULT, (void(*)(void))&matrixfree_apply<nvars>); 
	CHKERRQ(ierr);
	ierr = MatShellSetOperation(*A, MATOP_DESTROY, (void(*)(void))&matrixfree_destroy<nvars>); 
	CHKERRQ(ierr);

	ierr = MatSetUp(*A); CHKERRQ(ierr);
	return ierr;
}

template
StatusCode create_matrixfree_jacobian<NVARS>(const Spatial<freal,NVARS> *const s, Mat *const A);
template
StatusCode create_matrixfree_jacobian<1>(const Spatial<freal,1> *const s, Mat *const A);

bool isMatrixFree(Mat M) 
{
	MatType mattype;
	StatusCode ierr = MatGetType(M, &mattype);
	if(ierr != 0)
		throw "Could not get matrix type!";

	if(!strcmp(mattype,"shell"))
		return true;
	else
		return false;
}

/// Recursive function to return the first occurrence if a specific type of PC
StatusCode getPC(KSP ksp, const char *const type_name, PC* pcfound)
{
	StatusCode ierr = 0;
	PC pc;
	ierr = KSPGetPC(ksp, &pc); CHKERRQ(ierr);
	PetscBool isbjacobi, isasm, ismg, isgamg, isksp, isrequired;
	ierr = PetscObjectTypeCompare((PetscObject)pc,PCBJACOBI,&isbjacobi); CHKERRQ(ierr);
	ierr = PetscObjectTypeCompare((PetscObject)pc,PCASM,&isasm); CHKERRQ(ierr);
	ierr = PetscObjectTypeCompare((PetscObject)pc,PCMG,&ismg); CHKERRQ(ierr);
	ierr = PetscObjectTypeCompare((PetscObject)pc,PCGAMG,&isgamg); CHKERRQ(ierr);
	ierr = PetscObjectTypeCompare((PetscObject)pc,PCKSP,&isksp); CHKERRQ(ierr);
	ierr = PetscObjectTypeCompare((PetscObject)pc,type_name,&isrequired); CHKERRQ(ierr);

	if(isrequired) {
		// base case
		*pcfound = pc;
	}
	else if(isbjacobi || isasm)
	{
		PetscInt nlocalblocks, firstlocalblock;
		ierr = KSPSetUp(ksp); CHKERRQ(ierr); 
		ierr = PCSetUp(pc); CHKERRQ(ierr);
		KSP *subksp;
		if(isbjacobi) {
			ierr = PCBJacobiGetSubKSP(pc, &nlocalblocks, &firstlocalblock, &subksp); CHKERRQ(ierr);
		}
		else {
			ierr = PCASMGetSubKSP(pc, &nlocalblocks, &firstlocalblock, &subksp); CHKERRQ(ierr);
		}
		if(nlocalblocks != 1)
			SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, 
					"Only one subdomain per rank is supported.");
		ierr = getPC(subksp[0], type_name, pcfound); CHKERRQ(ierr);
	}
	else if(ismg || isgamg) {
		ierr = KSPSetUp(ksp); CHKERRQ(ierr); 
		ierr = PCSetUp(pc); CHKERRQ(ierr);
		PetscInt nlevels;
		ierr = PCMGGetLevels(pc, &nlevels); CHKERRQ(ierr);
		for(int ilvl = 0; ilvl < nlevels; ilvl++) {
			KSP smootherctx;
			ierr = PCMGGetSmoother(pc, ilvl , &smootherctx); CHKERRQ(ierr);
			ierr = getPC(smootherctx, type_name, pcfound); CHKERRQ(ierr);
		}
		KSP coarsesolver;
		ierr = PCMGGetCoarseSolve(pc, &coarsesolver); CHKERRQ(ierr);
		ierr = getPC(coarsesolver, type_name, pcfound); CHKERRQ(ierr);
	}
	else if(isksp) {
		ierr = KSPSetUp(ksp); CHKERRQ(ierr); 
		ierr = PCSetUp(pc); CHKERRQ(ierr);
		KSP subksp;
		ierr = PCKSPGetKSP(pc, &subksp); CHKERRQ(ierr);
		ierr = getPC(subksp, type_name, pcfound); CHKERRQ(ierr);
	}

	return ierr;
}

#ifdef USE_BLASTED

template <int nvars>
StatusCode setup_blasted(KSP ksp, Vec u, const Spatial<freal,nvars> *const startprob,
                         Blasted_data_list& bctx)
{
	StatusCode ierr = 0;
	Mat M, A;
	ierr = KSPGetOperators(ksp, &A, &M); CHKERRQ(ierr);

	// first assemble the matrix once because PETSc requires it
	ierr = startprob->assemble_jacobian(u, M); CHKERRQ(ierr);
	ierr = MatAssemblyBegin(M, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
	ierr = MatAssemblyEnd(M, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);

	ierr = setup_blasted_stack(ksp,&bctx); CHKERRQ(ierr);

	return ierr;
}

template StatusCode setup_blasted(KSP ksp, Vec u, const Spatial<freal,NVARS> *const startprob, 
                                  Blasted_data_list& bctx);
template StatusCode setup_blasted(KSP ksp, Vec u, const Spatial<freal,1> *const startprob, 
                                  Blasted_data_list& bctx);
#endif



// AB



PetscErrorCode MatrixFreePreconditioner:: getLU(Mat A) {

	StatusCode ierr = 0;
	// MatCopy(A,Lmat,SAME_NONZERO_PATTERN);
	// MatCopy(A,Umat,SAME_NONZERO_PATTERN);

	//MatConvert(A,MATSAME,MAT_INITIAL_MATRIX, &Lmat);
	//MatConvert(A,MATSAME,MAT_INITIAL_MATRIX, &Umat);
	//MatConvert(A,MATSAME,MAT_INITIAL_MATRIX, &D);

	ierr = MatDuplicate(A,MAT_COPY_VALUES,&Lmat);CHKERRQ(ierr);
	ierr = MatDuplicate(A,MAT_COPY_VALUES,&Umat);CHKERRQ(ierr);
	ierr = MatDuplicate(A,MAT_COPY_VALUES,&D);CHKERRQ(ierr);

	PetscInt blk_size; 
	PetscInt m;
	PetscInt n;

	MatGetBlockSize(A,&blk_size);
	MatGetSize(A, &m, &n); // get matrix size 

	int b = m/blk_size;
	PetscInt rows[blk_size];
	PetscInt cols[blk_size];
	PetscScalar Val[blk_size*blk_size];



	for (int i = 0; i < blk_size*blk_size; i++)
	{

			Val[i] = 0.0;
		
	}
	
	const PetscScalar *val1 = Val;
	for (int i = 0; i < b; i++)
	{

		int p = i*blk_size;
		for (int j = 0; j < blk_size; j++)
		{
			rows[j] = p+j;
			cols[j] = p+j;
		}

		// zero out the diagonal blocks
		const PetscInt *rows1 = rows;
		const PetscInt *cols1 = cols;
		MatSetValues(Lmat,blk_size,rows1,blk_size,cols1,val1,INSERT_VALUES);
		MatSetValues(Umat,blk_size,rows1,blk_size,cols1,val1,INSERT_VALUES);
	
		
		// Zero out the upper triangular blocks in Lmat and D

		for (int j = i+1; j < b; j++)
		{
			for (int k = 0; k<blk_size; k++)
			{
				cols[k] = cols[k] + blk_size;

			}
			
			const PetscInt *cols1 = cols;
			MatSetValues(Lmat,blk_size,rows1,blk_size,cols1,val1,INSERT_VALUES);
			MatSetValues(D,blk_size,rows1,blk_size,cols1,val1,INSERT_VALUES);
		}
		
		// zero out the lower triangle blocks in Umat and D
		for (int j = 0; j < blk_size; j++)
		{
			
			cols[j] = rows[j];
		}
		for (int j = i-1; j >=0; j--)
		{
			for (int k = 0; k<blk_size; k++)
			{
				cols[k] = cols[k] - blk_size;

			}
			
			const PetscInt *rows1 = rows;
			const PetscInt *cols1 = cols;
			MatSetValues(Umat,blk_size,rows1,blk_size,cols1,val1,INSERT_VALUES);
			MatSetValues(D,blk_size,rows1,blk_size,cols1,val1,INSERT_VALUES);
		}

	}

	MatAssemblyBegin(Lmat,MAT_FINAL_ASSEMBLY);
	MatAssemblyEnd(Lmat,MAT_FINAL_ASSEMBLY);
	MatAssemblyBegin(Umat,MAT_FINAL_ASSEMBLY);
	MatAssemblyEnd(Umat,MAT_FINAL_ASSEMBLY);
	MatAssemblyBegin(D,MAT_FINAL_ASSEMBLY);
	MatAssemblyEnd(D,MAT_FINAL_ASSEMBLY);


	//checking whether the facorization is good

	return 0;

}



PetscErrorCode MatrixFreePreconditioner:: nbgetLU(Mat A) {
	StatusCode ierr = 0;

	// MatCopy(A,Lmat,SAME_NONZERO_PATTERN);
	// MatCopy(A,Umat,SAME_NONZERO_PATTERN);

	//ierr =MatConvert(A,MATSAME,MAT_INITIAL_MATRIX, &Lmat);CHKERRQ(ierr);
	//ierr =MatConvert(A,MATSAME,MAT_INITIAL_MATRIX, &Umat);CHKERRQ(ierr);
	//ierr =MatConvert(A,MATSAME,MAT_INITIAL_MATRIX, &D);CHKERRQ(ierr);

	ierr = MatDuplicate(A,MAT_COPY_VALUES,&Lmat);CHKERRQ(ierr);
	ierr = MatDuplicate(A,MAT_COPY_VALUES,&Umat);CHKERRQ(ierr);
	ierr = MatDuplicate(A,MAT_COPY_VALUES,&D);CHKERRQ(ierr);

	PetscInt m;
	PetscInt n;

	MatGetSize(A, &m, &n); // get matrix size 

	
			
	for (PetscInt i = 0; i < m; i++)
	{
		for (PetscInt j = 0; j < m; j++)
		{
			if (i==j)
			{
				ierr =MatSetValue(Lmat,i,j,0.0, INSERT_VALUES);CHKERRQ(ierr);
				ierr =MatSetValue(Umat,i,j,0.0, INSERT_VALUES);CHKERRQ(ierr);
			}
			
			if (i<j)
			{
				ierr =MatSetValue(Lmat,i,j,0.0, INSERT_VALUES);CHKERRQ(ierr);
				ierr =MatSetValue(D,i,j,0.0, INSERT_VALUES);CHKERRQ(ierr);
			}		
			
			if (i>j)
			{
				ierr =MatSetValue(Umat,i,j,0.0, INSERT_VALUES);CHKERRQ(ierr);
				ierr =MatSetValue(D,i,j,0.0, INSERT_VALUES);CHKERRQ(ierr);
			}
		}
			
			
		
		
	}
	

	MatAssemblyBegin(Lmat,MAT_FINAL_ASSEMBLY);
	MatAssemblyEnd(Lmat,MAT_FINAL_ASSEMBLY);
	MatAssemblyBegin(Umat,MAT_FINAL_ASSEMBLY);
	MatAssemblyEnd(Umat,MAT_FINAL_ASSEMBLY);
	MatAssemblyBegin(D,MAT_FINAL_ASSEMBLY);
	MatAssemblyEnd(D,MAT_FINAL_ASSEMBLY);

	
	


	//checking whether the facorization is good
	Mat S; 
	ierr = MatDuplicate(Lmat,MAT_COPY_VALUES,&S);CHKERRQ(ierr);
	//ierr =MatConvert(Lmat,MATSAME,MAT_INITIAL_MATRIX, &S);CHKERRQ(ierr);
	ierr =MatAXPY(S,1,D, SAME_NONZERO_PATTERN);CHKERRQ(ierr);
	ierr =MatAXPY(S,1,Umat, SAME_NONZERO_PATTERN);CHKERRQ(ierr);
	ierr =MatAXPY(S,-1,A, SAME_NONZERO_PATTERN);CHKERRQ(ierr);
	PetscReal nrm;
	ierr =MatNorm(S,NORM_FROBENIUS,&nrm);CHKERRQ(ierr);

	
	if (nrm <=1e-6)
	{
		std::cout<<nrm<< "Hehe"<<std::endl;
	}
	else
	{
		std::cout<<nrm<< "Not Hehe"<<std::endl;

	}
	 


	return 0;
}




	//template <int nvars>
	//StatusCode MatrixFreePreconditioner::
	PetscErrorCode mf_pc_create(MatrixFreePreconditioner **shell){
	// Set up matrix free PC
	StatusCode ierr = 0;
	MatrixFreePreconditioner *newctx;
	ierr = PetscNew(&newctx);CHKERRQ(ierr);
	// newctx->diag = 0;
	*shell = newctx;
	return 0;
	}
	

	// template <int nvars>
	// StatusCode MatrixFreePreconditioner::
	PetscErrorCode mf_pc_setup(PC pc){
	// Set up matrix free PC
	StatusCode ierr = 0;
	MatrixFreePreconditioner *shell;
	ierr = PCShellGetContext(pc,&shell);CHKERRQ(ierr);
	Mat A;
	ierr= PCGetOperators(pc,NULL,&A);
	ierr = MatDuplicate(A,MAT_DO_NOT_COPY_VALUES,&(shell->Dinv)); CHKERRQ(ierr);

	//ierr = MatConvert(A,MATSAME,MAT_INITIAL_MATRIX, &(shell->Dinv)); CHKERRQ(ierr);
	ierr = MatScale(shell->Dinv,0); CHKERRQ(ierr);
	//ierr = MatCreate(PETSC_COMM_SELF,&(shell->Dinv));
	ierr = MatInvertBlockDiagonalMat(A,shell->Dinv); CHKERRQ(ierr);
	shell->getLU(A);


	
	/*PetscInt m;
	PetscInt n;
	MatGetSize(A, &m, &n);
	VecCreate(PETSC_COMM_WORLD,&(shell->diag));
	VecSetSizes(shell->diag,PETSC_DECIDE,m);
    MatGetDiagonal(A,shell->diag);
    VecReciprocal(shell->diag);
    //shell->diag = diag1;*/
	return 0;

	}

	//template <int nvars>
	//StatusCode MatrixFreePreconditioner::
	PetscErrorCode mf_pc_apply(PC pc, Vec x, Vec y){
	// Set up matrix free PC
	// MatrixFreePreconditioner mfp;
		PetscInt ierr = 0;
		MatrixFreePreconditioner *shell;
		ierr = PCShellGetContext(pc,&shell);CHKERRQ(ierr);
		//mc_lusgs(x,y);
		Vec y1;
		Vec y2,y3,temp;
		VecDuplicate(x,&y1);
		VecDuplicate(x,&y2);
		VecDuplicate(x,&y3);
		VecDuplicate(x,&temp);

		VecSet(y1,0);
		VecAssemblyBegin(y1);
		VecAssemblyEnd(y1);

		//fvens::MatrixFreePreconditioner *shell;
			
		
		PetscReal tol1 = 10;

		while (tol1>1e-3)
		{
			VecCopy(y2,y1);

			// y1 = Dinv(x-Lmat*y1);
			MatMult(shell->Lmat,y1,temp); //temp = Lmat*y1
			VecAYPX(temp,-1,x);//temp = x-Lmat*y1
			MatMult(shell->Dinv,temp,y1); //Dinv(x-Lmat*y1)
			//VecPointwiseMult(y1,x,shell->diag);

			// error tol to check convergence
			VecAXPY(y1,-1,y2);
			VecNorm(y1,NORM_2,&tol1);
			std::cout<<tol1<<"tol1"<<std::endl;

		}

		PetscReal tol = 10;
		VecCopy(y1,y);
		while (tol>1e-3)
		{
			VecCopy(y,y3);

			//y = y1 - Dinv * Umat * y
			MatMult(shell->Umat,y,temp); // temp = Umat * y
			//VecPointwiseMult(y,x,shell->diag);
			MatMult(shell->Dinv,temp,y); //y = Dinv * Umat * y
			VecAYPX(y,-1,y1); //y = y1 - Dinv * Umat * y

			// Residual to compute tolerance
			VecAXPY(y,-1,y3);
			VecNorm(y,NORM_2,&tol);
			
			//Storing the old vectors
			
			std::cout<<tol<<std::endl;
		}
		return 0;
		

	}

	//template <int nvars>
	//StatusCode MatrixFreePreconditioner::
	PetscErrorCode mf_pc_destroy(PC pc){
	// Set up matrix free PC
	StatusCode ierr = 0;
	// Vec diag;

	MatrixFreePreconditioner *shell;
	ierr = PCShellGetContext(pc,&shell);CHKERRQ(ierr);
	ierr = MatDestroy(&shell->Lmat);CHKERRQ(ierr);
	ierr = MatDestroy(&shell->Umat);CHKERRQ(ierr);
	ierr = MatDestroy(&shell->Dinv);CHKERRQ(ierr);
	ierr = PetscFree(shell);CHKERRQ(ierr);
	return 0;

	}

}







/*
PetscErrorCode mc_lusgs(Vec x, Vec y){

	//Matrix A to apply LU-SGS in matrix format. 

	// Get the blocks of the matrix (D, L, U)


	Vec y1;
	Vec y2;
	VecDuplicate(x,&y1);
	VecDuplicate(x,&y2);

	VecSet(y,0);
	VecSet(y1,0);
	VecSet(y2,0);

	fvens::MatrixFreePreconditioner *shell;
	fvens::LU_dat lu;

	PetscReal tol = 1e-3;

	while (tol>1e-3)
	{
		Vec temp;
		VecDuplicate(x,&temp);

		// y1 = Dinv(x-Lmat*y1);
		MatMult(lu.Lmat,y1,temp);
		VecAXPY(temp,-1,x);
		MatMult(shell->Dinv,temp,y1);

		//y = y1 - Dinv * Umat * y
		MatMult(lu.Umat,y,temp);
		MatMult(shell->Dinv,temp,y);
		VecAXPY(y,-1,y1);

		// Residual to compute tolerance
		VecAXPY(y,-1,y2);
		VecNorm(y,NORM_2,&tol);
		
		//Storing the old vectors
		VecCopy(y,y2);


	}

	return 0;
	
}*/




	/*
	Vec d; //diagonal entries of the matrix
	PetscInt m,n;
	VecDuplicate(x,&yst);
	VecDuplicate(x,&d);


	MatGetDiagonal(A, d);
	MatGetSize(A,&m,&n);
	VecReciprocal(d);

	//
	 Mat Dinv; // Matrix with inverse diagonal blocks of A
	//MatDuplicate(A,MAT_DO_NOT_COPY_VALUES,&Dinv);
	//MatInvertBlockDiagonalMat(A,&Dinv);

	int nblock = n/NVARS; 

	for (int i = 0;i<nblock;i++){



	} */



