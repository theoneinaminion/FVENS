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
	// std::cout<<"eps:"<<eps<<std::endl;
	// VecDuplicate(u_state, &u);
	// VecDuplicate(r_state, &res);
	// VecDuplicate(dtms, &mdt);

	// VecCopy(u_state, u);
	// VecCopy(r_state, res);
	// VecCopy(dtms, mdt);
	std::cout << " Matfree state set" << std::endl;
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


template<int nvars>
PetscErrorCode MatrixFreePreconditioner<nvars>:: getLU(Mat A) {

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



template<int nvars>
PetscErrorCode MatrixFreePreconditioner<nvars>:: nbgetLU(Mat A) {
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


template<int nvars>
double MatrixFreePreconditioner<nvars>:: epsilon_calc(Vec x, Vec y) {

		double eps; 
		PetscInt vecsize;
		int ierr; 
		PetscScalar nrm1, nrm2; 
		ierr = VecGetSize (x,&vecsize);CHKERRQ(ierr);
		ierr = VecNorm (x,NORM_1,&nrm1);CHKERRQ(ierr);
		ierr = VecNorm (y,NORM_2,&nrm2);CHKERRQ(ierr);
				
		eps = nrm1/(vecsize*nrm2);
		eps = eps*(std::pow(10,-6))+std::pow(10,-6); // epsilon used in matrix-free finite diff
		
		double pertmag = (1e-6)/nrm2;
		return pertmag;

}




	//template <int nvars>
	//StatusCode MatrixFreePreconditioner::
	template<int nvars>
	PetscErrorCode mf_pc_create(MatrixFreePreconditioner<nvars> **shell){
	// Set up matrix free PC
	StatusCode ierr = 0;
	MatrixFreePreconditioner<nvars> *newctx;
	ierr = PetscNew(&newctx);CHKERRQ(ierr);

	*shell = newctx;
	return 0;
	}
	
	template
	PetscErrorCode mf_pc_create<NVARS>(MatrixFreePreconditioner<NVARS> **shell);
	template
	PetscErrorCode mf_pc_create<1>(MatrixFreePreconditioner<1> **shell);

	// template <int nvars>
	// StatusCode MatrixFreePreconditioner::
	template<int nvars>
	PetscErrorCode mf_pc_setup(PC pc, Vec u, Vec r,const Spatial<freal,nvars> *const space, MatrixFreePreconditioner<nvars> *shell){
	// Set up matrix free PC
	StatusCode ierr = 0;
	//MatrixFreePreconditioner *shell;
	ierr = PCShellGetContext(pc,shell);CHKERRQ(ierr);
	Mat A;
	ierr= PCGetOperators(pc,NULL,&A);CHKERRQ(ierr);
	ierr = VecDuplicate(u,&(shell->uvec));CHKERRQ(ierr);
	ierr = VecDuplicate(r,&(shell->rvec));CHKERRQ(ierr);
	ierr = VecCopy(u,shell->uvec);CHKERRQ(ierr);
	ierr = VecCopy(r,shell->rvec);CHKERRQ(ierr);

	ierr = MatDuplicate(A,MAT_DO_NOT_COPY_VALUES,&(shell->Dinv)); CHKERRQ(ierr);
	ierr = MatGetBlockSize(A,&(shell->blk_size));CHKERRQ(ierr);
	ierr = MatGetSize(A, &(shell->m), &(shell->n));CHKERRQ(ierr); // get matrix size  
	shell->space = space; 

	ierr = MatScale(shell->Dinv,0); CHKERRQ(ierr);
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

	template
	PetscErrorCode mf_pc_setup<NVARS>(PC pc, Vec u, Vec r,const Spatial<freal,NVARS> *const space, MatrixFreePreconditioner<NVARS> *shell);
	template
	PetscErrorCode mf_pc_setup<1>(PC pc, Vec u, Vec r,const Spatial<freal,1> *const space, MatrixFreePreconditioner<1> *shell);
	

	//template <int nvars>
	//StatusCode MatrixFreePreconditioner::
	template<int nvars>
	PetscErrorCode mf_pc_apply(PC pc, Vec x, Vec y){
	// Set up matrix free PC
	// MatrixFreePreconditioner mfp;

	
		PetscInt ierr = 0;
	
		MatrixFreePreconditioner<nvars> *shell;
		ierr = PCShellGetContext(pc,&shell);CHKERRQ(ierr);

		//mc_lusgs(x,y);
		Vec temp,nrm,ust, rst,u,r, blank, yst, ycopy,yst2;

	// 	PetscInt locnelem;
	// ierr = VecGetSize(x, &locnelem); CHKERRQ(ierr);
	// std::cout<<locnelem<<std::endl;
	// ierr = VecGetSize(shell->uvec, &locnelem); CHKERRQ(ierr);
	// std::cout<<locnelem<<std::endl;
		
		ierr = VecDuplicate(shell->uvec,&nrm);CHKERRQ(ierr);
		ierr = VecDuplicate(shell->uvec,&ust);CHKERRQ(ierr);
		ierr = VecDuplicate(shell->uvec,&rst);CHKERRQ(ierr);
		ierr = VecDuplicate(shell->uvec,&blank);CHKERRQ(ierr);

		ierr = VecDuplicate(shell->uvec,&u);CHKERRQ(ierr);
		ierr = VecDuplicate(shell->uvec,&r);CHKERRQ(ierr);
		ierr = VecDuplicate(shell->uvec,&yst);CHKERRQ(ierr);
		ierr = VecDuplicate(shell->uvec,&ycopy);CHKERRQ(ierr);
		ierr = VecDuplicate(shell->uvec,&yst2);CHKERRQ(ierr);


	
		ierr = VecSet(blank,0);CHKERRQ(ierr);


		// matfree
		double eps;
		PetscInt vecsize;
		ierr = VecCopy(x,yst); // initialize yst
		ierr = VecCopy(x,y);CHKERRQ(ierr); // initialize y

		//set up ith block dinv matrix
		Mat Dinv_i;  //MatType type;
		ierr = MatCreateSeqAIJ(PETSC_COMM_WORLD, shell->blk_size, shell->blk_size, 0, NULL, &Dinv_i);CHKERRQ(ierr);
		ierr = MatSetOption(Dinv_i, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE);CHKERRQ(ierr);
		// ierr = MatCreate(PETSC_COMM_WORLD, &Dinv_i);CHKERRQ(ierr);
		// ierr = MatGetType(shell->Dinv, &type);CHKERRQ(ierr);
		// ierr = MatSetType(Dinv_i, type);CHKERRQ(ierr);
		// ierr = MatSetSizes(Dinv_i, PETSC_DECIDE, PETSC_DECIDE, shell->blk_size, shell->blk_size);


		// vector to store vector sums
		Vec sum, y_stari, y_i;
		ierr = VecCreate(PETSC_COMM_SELF, &sum);CHKERRQ(ierr);
		ierr = VecSetType(sum, VECMPI);CHKERRQ(ierr);
		ierr = VecSetSizes(sum, shell->blk_size, PETSC_DECIDE);CHKERRQ(ierr);
		ierr = VecDuplicate(sum,&temp);CHKERRQ(ierr);
		ierr = VecDuplicate(sum,&y_stari);CHKERRQ(ierr);
		ierr = VecDuplicate(sum,&y_i);CHKERRQ(ierr);
		
		PetscInt b = (shell->n)/(shell->blk_size);
		// std::cout<<"b"<<b<<std::endl;
		// std::cout<<"n"<<shell->n<<std::endl;
		// std::cout<<"blk_size"<<shell->blk_size<<std::endl;
		
		int maxiter = 10;
		int iter = 0;
	    PetscReal tol1 = 10;
		while (tol1>=1e-6 && iter<=maxiter)
		{	

			ierr = VecCopy(y,ycopy);CHKERRQ(ierr); // Saving the value from prev iteration
			iter = iter+1;
			std::cout<<"iter"<<iter<<std::endl;
			
			ierr = VecGetSize (yst,&vecsize);CHKERRQ(ierr);
			//std::cout<<vecsize<<std::endl;
				
			
			// Looping over the elements

			//PetscInt size;
			// ierr = VecGetLocalSize(shell->rvec, &size); CHKERRQ(ierr);
			 //std::cout<<nvars<<std::endl;
			for (int i = 0; i < b; i++)
			{

				
				eps = shell->epsilon_calc(shell->uvec, yst);
				ierr = VecWAXPY(ust,eps,yst,shell->uvec);CHKERRQ(ierr); // ust = eps*yst + shell->uvec
				ierr = shell->space->compute_residual(ust, rst, false, blank); CHKERRQ(ierr); // r(u+eps*yst)
				//std::cout<<eps<<std::endl;
				ierr = VecAssemblyBegin(rst);CHKERRQ(ierr);
				ierr = VecAssemblyEnd(rst);CHKERRQ(ierr);

				

				eps  =  shell->epsilon_calc(shell->uvec, y);
				ierr = VecWAXPY(u,eps,y,shell->uvec);CHKERRQ(ierr); // ust = eps*yst + shell->uvec
				ierr = shell->space->compute_residual(u, r, false, blank); CHKERRQ(ierr); // r(u+eps*yst)
				//std::cout<<eps<<std::endl;
				ierr = VecAssemblyBegin(r);CHKERRQ(ierr);
				ierr = VecAssemblyEnd(r);CHKERRQ(ierr);	
				
				
					
				
				ierr = VecSet(sum,0);CHKERRQ(ierr);
								
				PetscScalar val,va, rsty[shell->blk_size], ry[shell->blk_size];
				PetscInt row, col,idx[shell->blk_size];

				// L loop
				for (int j = 0; j <= i; j++)
				{					
					//std::cout<<i<<"--"<<j<<std::endl;
					
					for (int k = 0; k < shell->blk_size; k++)
					{
						idx[k] = nvars*j+k; // Indices of values in a given block
						//std::cout<<nvars<<"NVARS"<<std::endl;
					}

					// get respective values of r(w_j+eps*yst_j) and r(w_j) in the given block j.
					ierr = VecGetValues(shell->rvec,shell->blk_size,idx,ry);CHKERRQ(ierr);
					ierr = VecGetValues(rst,shell->blk_size,idx,rsty);CHKERRQ(ierr);
					
					
					for (PetscInt k = 0; k < shell->blk_size; k++)
					{	
						val = (rsty[k]-ry[k])/eps;
						ierr = VecSetValue(sum,k,val,ADD_VALUES); CHKERRQ(ierr);
					}
					ierr = VecAssemblyBegin(sum);CHKERRQ(ierr);
					ierr = VecAssemblyEnd(sum);CHKERRQ(ierr);


				}
					
				//PetscInt indx[shell->blk_size];
					
				PetscScalar xget[shell->blk_size],sum_get;
				ierr = VecGetValues(x,shell->blk_size,idx,xget);CHKERRQ(ierr); // Changed idx to indx. Check if it is logically correct. 
				

				for (PetscInt k = 0; k < shell->blk_size; k++)
				{	
					ierr = VecGetValues(sum,1,&k,&sum_get);CHKERRQ(ierr);
					val = xget[k]-sum_get;
					ierr = VecSetValue(temp,k,val,INSERT_VALUES); CHKERRQ(ierr); 
				}
				ierr = VecAssemblyBegin(temp);CHKERRQ(ierr);
				ierr = VecAssemblyEnd(temp);CHKERRQ(ierr);

				//  Dinv_{i block}*temp = yst_jblock here 
				
				for (PetscInt k = 0; k < shell->blk_size; k++)
				{
					 row = i*(shell->blk_size)+k;
					
					for (PetscInt l = 0; l < shell->blk_size; l++)
					{
						 col = i*(shell->blk_size)+l;
						
						ierr = MatGetValue(shell->Dinv, row, col, &va); CHKERRQ(ierr); 
						ierr = MatSetValue(Dinv_i,k,l,va,INSERT_VALUES); CHKERRQ(ierr); 
						//std::cout<<"--"<<row<<col<<std::endl;

					}
					
					
					 idx[k] = row; // why am I storing idx? 2023-05-18
				}
				ierr = MatAssemblyBegin(Dinv_i,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr); 
				ierr = MatAssemblyEnd(Dinv_i,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

				ierr = MatMult(Dinv_i, temp, y_stari);CHKERRQ(ierr); //temporarily store the product in sum
				
				
				// writing the final values to yst

				for (PetscInt k = 0; k < shell->blk_size; k++)
				{
					ierr = VecGetValues(y_stari,1,&k,&va);CHKERRQ(ierr);
					row =  i*(shell->blk_size)+k;
					ierr = VecSetValue(yst, row, va, INSERT_VALUES);CHKERRQ(ierr);
				}
				ierr = VecAssemblyBegin(yst);CHKERRQ(ierr);
				ierr = VecAssemblyEnd(yst);CHKERRQ(ierr); // \Delta U^* has been assembled for the entire domain. 

				


				// U loop 
				// This is the problem. 2023-11-16
				ierr = VecSet(sum,0); CHKERRQ(ierr); 
				for (int j = i; j < b; j++)
				{
					for (int k = 0; k < shell->blk_size; k++)
					{
						idx[k] = nvars*j+k; 
						//std::cout<<idx[k]<<"idx"<<std::endl;
					}

					//ierr = VecGetSize (shell->rvec,&vecsize);CHKERRQ(ierr);
					//std::cout<<vecsize<<"vecsize"<<std::endl;
					// get respective values of r(w_j+eps*z_j) and r(w_j) in he given block.
					ierr = VecGetValues(shell->rvec,shell->blk_size,idx,ry);CHKERRQ(ierr);
					ierr = VecGetValues(r,shell->blk_size,idx,rsty);CHKERRQ(ierr); // This is wrong. The new residual must have updated yst. Is that what is going on? check. 2023-11-16
					
					
					for (PetscInt k = 0; k < shell->blk_size; k++)
					{	
						val = (rsty[k]-ry[k])/eps;

						if (val > 1000) {std::cout<<"val------"<<val<<std::endl;
						std::cout<<"i------"<<i<<std::endl;
						std::cout<<"j-----"<<j<<std::endl;
						std::cout<<"rsty------"<<rsty[k]<<std::endl;
						std::cout<<"ry------"<<ry[k]<<std::endl;
						return -1;
						}
						
						ierr = VecSetValue(sum,k,val,ADD_VALUES); CHKERRQ(ierr);
					}
					ierr = VecAssemblyBegin(sum);CHKERRQ(ierr);
					ierr = VecAssemblyEnd(sum);CHKERRQ(ierr);
				}
				
				ierr = MatMult(Dinv_i, sum, temp);CHKERRQ(ierr); 
				ierr = VecWAXPY(y_i,-1,temp,y_stari); // y_i = y_stari - temp
				
				
				// if (i==1)
				// {
				// 	writePetscObj(sum, "sum");
				// 	return -1;
					
				// }	



				for (PetscInt k = 0; k < shell->blk_size; k++)
				{
					ierr = VecGetValues(y_i,1,&k,&va);CHKERRQ(ierr);
					row =  i*(shell->blk_size)+k;
					ierr = VecSetValue(y, row, va, INSERT_VALUES);CHKERRQ(ierr);
				}
				ierr = VecAssemblyBegin(y);CHKERRQ(ierr);
				ierr = VecAssemblyEnd(y);CHKERRQ(ierr); 

				// if (i==(b-1))
				// {
				// 	int fs = 1;
				// 	std::cout<<fs<< "i loop"<<std::endl;
				// 	fs = fs+1;
				// }
			
				
			}




			// error tol to check convergence
			ierr = VecWAXPY(nrm,-1.0,y,ycopy);CHKERRQ(ierr);
			ierr =VecNorm(nrm,NORM_2,&tol1);CHKERRQ(ierr);
			std::cout<<tol1<<"tol1"<<std::endl;

		}
		return ierr;

	}

	template
	PetscErrorCode mf_pc_apply<NVARS>(PC pc, Vec x, Vec y);
	template
	PetscErrorCode mf_pc_apply<1>(PC pc, Vec x, Vec y);



	
	# if 0
	template<int nvars>
	PetscErrorCode mf_pc_apply1(PC pc,const Vec x, Vec y)
	{
		StatusCode ierr = 0;
		MatrixFreePreconditioner<nvars> *shell;
		ierr = PCShellGetContext(pc,&shell);CHKERRQ(ierr);
		
		Vec sum, temp;
		ierr = VecCreate(PETSC_COMM_SELF, &sum);CHKERRQ(ierr);
		ierr = VecSetType(sum, VECMPI);CHKERRQ(ierr);
		ierr = VecSetSizes(sum, nvars, PETSC_DECIDE);CHKERRQ(ierr);
		ierr = VecDuplicate(sum,&temp);CHKERRQ(ierr);
		ierr = VecSet(sum,0);CHKERRQ(ierr);

		Vec yst; 
		ierr = VecDuplicate(shell->uvec,&yst);CHKERRQ(ierr);
		ierr = VecCopy(x,yst);CHKERRQ(ierr); // initialize yst
		ierr = VecCopy(x,y);CHKERRQ(ierr); // initialize y

		Mat Dinv_i;  //MatType type;
		ierr = MatCreateSeqAIJ(PETSC_COMM_WORLD, shell->blk_size, shell->blk_size, 0, NULL, &Dinv_i);CHKERRQ(ierr);
		ierr = MatSetOption(Dinv_i, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE);CHKERRQ(ierr);

		Vec dummy = NULL;

		if(!(shell->space))
			SETERRQ(PETSC_COMM_SELF, PETSC_ERR_POINTER,
					"Spatial context not set!");

		const UMesh<freal,NDIM> *const m = shell->space->mesh();
		//ierr = VecSet(y, 0.0); CHKERRQ(ierr);

		Vec auxl, rfd_l,auxu,rfd_u;
		ierr = VecDuplicate(shell->uvec, &auxl); CHKERRQ(ierr);
		ierr = VecDuplicate(shell->rvec, &rfd_l); CHKERRQ(ierr);
		ierr = VecDuplicate(shell->uvec, &auxu); CHKERRQ(ierr);
		ierr = VecDuplicate(shell->rvec, &rfd_u); CHKERRQ(ierr);

		PetscScalar xnorm = 0;
		ierr = VecNorm(x, NORM_2, &xnorm); CHKERRQ(ierr);

	#ifdef DEBUG
		if(xnorm < 10.0*std::numeric_limits<freal>::epsilon())
			SETERRQ(PETSC_COMM_SELF, PETSC_ERR_FP,
					"Norm of offset is too small for finite difference Jacobian!");
	#endif

		for (fint iel = 0; iel < m->gnelem()*nvars; iel++) 
		{
				PetscInt row,col;
				PetscScalar va;
				// set up Dinv matrix for the current element
				for (PetscInt k = 0; k < nvars; k++)
				{
					 row = iel*nvars+k;
					
					for (PetscInt l = 0; l < nvars; l++)
					{
						col = iel*nvars+l;
						
						ierr = MatGetValue(shell->Dinv, row, col, &va); CHKERRQ(ierr); 
						ierr = MatSetValue(Dinv_i,k,l,va,INSERT_VALUES); CHKERRQ(ierr); 
						//std::cout<<"--"<<row<<col<<std::endl;

					}
					
				}				
		
				ierr = MatAssemblyBegin(Dinv_i,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr); 
				ierr = MatAssemblyEnd(Dinv_i,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);



				const freal eps = 1e-6;
				const freal pertmag = eps/xnorm;


				// aux <- u + eps/xnorm * x ;    y <- 0
				{
					ConstVecHandler<PetscScalar> uh(shell->uvec);
					const PetscScalar *const u_arr = uh.getArray();
					MutableVecHandler<PetscScalar> auxh(auxl);
					PetscScalar *const aux_l = auxh.getArray(); 
					MutableVecHandler<PetscScalar> auxi(auxu);
					PetscScalar *const aux_u = auxi.getArray();
					MutableVecHandler<PetscScalar> ygl(rfd_l);
					PetscScalar *const res_l = ygl.getArray();
					MutableVecHandler<PetscScalar> ygu(rfd_u);
					PetscScalar *const res_u = ygu.getArray();
					MutableVecHandler<PetscScalar> ysth(yst);
					PetscScalar *const yst_arr = ysth.getArray();
					MutableVecHandler<PetscScalar> ygh(y);
					PetscScalar *const y_arr = ygh.getArray();

			//#pragma omp parallel for simd default(shared)
					for(fint i = 0; i < m->gnelem()*nvars; i++) 
					{
						res_l[i] = 0;
						res_u[i] = 0;
						aux_l[i] = u_arr[i] + pertmag * yst_arr[i];
						aux_u[i] = u_arr[i] + pertmag * y_arr[i];
					}
					
				}

				// ierr = VecGhostUpdateBegin(auxl, INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);
				// ierr = VecGhostUpdateEnd(auxl, INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);

				// ierr = VecGhostUpdateBegin(auxu, INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);
				// ierr = VecGhostUpdateEnd(auxu, INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);

				// y <- -r(u + eps/xnorm * x)
				ierr = shell->space->compute_residual(auxl, rfd_l, false, dummy); CHKERRQ(ierr);

				// ierr = VecGhostUpdateBegin(rfd_l, INSERT_VALUES, SCATTER_REVERSE); CHKERRQ(ierr);
				// ierr = VecGhostUpdateEnd(rfd_l, INSERT_VALUES, SCATTER_REVERSE); CHKERRQ(ierr);

				ierr = shell->space->compute_residual(auxu, rfd_u, false, dummy); CHKERRQ(ierr);

				// ierr = VecGhostUpdateBegin(rfd_u, INSERT_VALUES, SCATTER_REVERSE); CHKERRQ(ierr);
				// ierr = VecGhostUpdateEnd(rfd_u, INSERT_VALUES, SCATTER_REVERSE); CHKERRQ(ierr);
				
				// y <- vol/dt x + (-(-r(u + eps/xnorm * x)) + (-r(u))) / eps |x|
				//    = vol/dt x + (r(u + eps/xnorm * x) - r(u)) / eps |x|
				/* We need to divide the difference by the step length scaled by the norm of x.
				* We do NOT divide by epsilon, because we want the product of the Jacobian and x, which is
				* the directional derivative (in the direction of x) multiplied by the norm of x.
				*/
				

				{
					ConstVecHandler<PetscScalar> xh(x);
					const PetscScalar *const x_arr = xh.getArray();
					ConstVecHandler<PetscScalar> resh(shell->rvec);
					const PetscScalar *const res_arr = resh.getArray();
					MutableVecHandler<PetscScalar> ygl(rfd_l);
					PetscScalar *const res_l = ygl.getArray();
					MutableVecHandler<PetscScalar> ygu(rfd_u);
					PetscScalar *const res_u = ygu.getArray();
					MutableVecHandler<PetscScalar> yi(yst);
					PetscScalar *const yst_arr = yi.getArray();
					MutableVecHandler<PetscScalar> yh(y);
					PetscScalar *const y_arr = yh.getArray();
					MutableVecHandler<PetscScalar> sumh(sum);
					PetscScalar *const sum_l = sumh.getArray(); 
					


					// L-loop
					//setting the sum to be equal to zero. 
			//#pragma omp parallel for simd default(shared)
					for(fint jel = 0; jel < iel; jel++)
					{
						for(int k = 0; k < nvars; k++) {
							// finally, add the pseudo-time term (Vol/dt du = Vol/dt x)
							sum_l[k] = sum_l[k] + (res_l[jel*nvars+k] - res_arr[jel*nvars+k])/pertmag;
						}

						for(int k = 0; k < nvars; k++) {
							sum_l[k] = x_arr[jel*nvars+k] - sum_l[k]; // sum = x-\sigma_{jel=0:iel-1} (r(u+eps*x)-r(u))/eps
						}
					}

					//Do the D^{-1}*sum
					ierr = MatMult(Dinv_i, sum, temp);CHKERRQ(ierr);
						for (fint k = 0; k < nvars; k++)
						{
							ierr = VecGetValues(temp,1,&k,&va);CHKERRQ(ierr);
							yst_arr[iel*nvars + k] = va;
						}
						
					

					ierr = VecSet(sum,0);CHKERRQ(ierr);
					MutableVecHandler<PetscScalar> sumi(sum);
					PetscScalar *const sum_u = sumi.getArray();

					// U-loop
			//#pragma omp parallel for simd default(shared)		
					for(fint jel = iel+1; jel < m->gnelem()*nvars; jel++)
					{
						for(int k = 0; k < nvars; k++) {
							// finally, add the pseudo-time term (Vol/dt du = Vol/dt x)
							sum_u[k] = sum_u[k]+(res_u[jel*nvars+k] - res_arr[jel*nvars+k])/pertmag;
						}
	
					}

					ierr = MatMult(Dinv_i, sum, temp);CHKERRQ(ierr);
						for (fint k = 0; k < nvars; k++)
						{
							ierr = VecGetValues(temp,1,&k,&va);CHKERRQ(ierr);
							y_arr[iel*nvars + k] = yst_arr[iel*nvars + k] - va;
						}


				}

		}


		return ierr;
	}
	

	template
	PetscErrorCode mf_pc_apply1<NVARS>(PC pc, const Vec x, Vec y);
	template
	PetscErrorCode mf_pc_apply1<1>(PC pc, const Vec x, Vec y);
	# endif

	template<int nvars>
	PetscErrorCode mf_pc_apply2(PC pc, Vec x, Vec y){

		int ierr = 0;
		MatrixFreePreconditioner<nvars> *shell;
		ierr = PCShellGetContext(pc,&shell);CHKERRQ(ierr);
		int nelem = (shell->n)/nvars;
		PetscScalar tol = 1e-6, nrm = 10;

		Vec z,yold, diff;
		ierr = VecDuplicate(shell->uvec,&z);CHKERRQ(ierr);
		ierr = VecDuplicate(shell->uvec,&yold);CHKERRQ(ierr);
		ierr = VecDuplicate(shell->uvec,&diff);CHKERRQ(ierr); // difference between y and yold
		ierr = VecCopy(x,z);CHKERRQ(ierr); // initialize z
		ierr = VecCopy(x,y);CHKERRQ(ierr); // initialize y
		
		/*
			// Initializing z and y to random vectors 
			init_rand(z);
			init_rand(y);

		*/

		// ####### Setup Auxillary data ################
		Vec sum, zelem, tempvec; // summing over the residuals

		ierr = VecCreate(PETSC_COMM_SELF, &sum);CHKERRQ(ierr);
		ierr = VecSetType(sum, VECMPI);CHKERRQ(ierr);
		ierr = VecSetSizes(sum, nvars, PETSC_DECIDE);CHKERRQ(ierr);
		ierr = VecDuplicate(sum,&zelem);CHKERRQ(ierr);
		ierr = VecDuplicate(sum,&tempvec);CHKERRQ(ierr);

		Mat Dinv_i;  //Inverse diagnoal at i^th element
		ierr = MatCreateSeqAIJ(PETSC_COMM_WORLD, nvars,nvars, 0, NULL, &Dinv_i);CHKERRQ(ierr);
		ierr = MatSetOption(Dinv_i, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE);CHKERRQ(ierr);

		Vec &rvec = shell->rvec; // residual vector r(shell->uvec) reference pointer

		PetscScalar relem[nvars], relem_pert[nvars], tempelem[nvars],val; 
		PetscInt idx[nvars];


		// ########### Start Iteration #################
		int iter = 0;
		while (nrm > tol)
		{
			ierr = 	VecCopy(y,yold); CHKERRQ(ierr);
			// writePetscObj(yold,"yold");
			// writePetscObj(x,"x");

			for (int i = 0; i < nelem; i++)
			{
				// #### Write the residual with u = shell->uvec + pertmag*z for L-Loop #####
				PetscScalar pertmag = shell->epsilon_calc(shell->uvec, z);
				Vec uvec_Lpert,rvec_L;
				ierr = VecDuplicate(shell->uvec,&rvec_L);CHKERRQ(ierr);
				ierr = VecDuplicate(shell->uvec,&uvec_Lpert);CHKERRQ(ierr);
				ierr = VecWAXPY(uvec_Lpert,pertmag,z,shell->uvec);CHKERRQ(ierr);
				shell->space->compute_residual(uvec_Lpert, rvec_L, false, NULL); CHKERRQ(ierr);

				ierr = VecGhostUpdateBegin(rvec_L, ADD_VALUES, SCATTER_REVERSE); CHKERRQ(ierr);
				ierr = VecGhostUpdateEnd(rvec_L, ADD_VALUES, SCATTER_REVERSE); CHKERRQ(ierr);

				// ############# Get the Diagonal Block of i^th element #############
				for (PetscInt k = 0; k < nvars; k++)
				{
					PetscInt row = i*nvars+k;
					
					for (PetscInt l = 0; l < nvars; l++)
					{
						PetscInt col = i*nvars+l;
						
						ierr = MatGetValue(shell->Dinv, row, col, &val); CHKERRQ(ierr); 
						ierr = MatSetValue(Dinv_i,k,l,val,INSERT_VALUES); CHKERRQ(ierr); 

					}
					idx[k] = row;
				}
				ierr = MatAssemblyBegin(Dinv_i,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr); 
				ierr = MatAssemblyEnd(Dinv_i,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

				// ############## Get x_i and store it in tempvec ######################
				ierr = VecGetValues(x,nvars,idx,tempelem);CHKERRQ(ierr); 

				for (int k = 0; k < nvars; k++)
				{
					idx[k] = k; //change idx coz tempvec is of the size 4
				}
				ierr = VecSetValues(tempvec,nvars,idx,tempelem,INSERT_VALUES);CHKERRQ(ierr);
				ierr = VecAssemblyBegin(tempvec);CHKERRQ(ierr);
				ierr = VecAssemblyEnd(tempvec);CHKERRQ(ierr);

				// #################    L-Loop	#####################
				ierr = VecSet(sum,0); CHKERRQ(ierr); // Initialize sum to 0
				for (int j = 0; j < i; j++)
				{ //std::cout<<"i,j-------"<<i<<"  "<<j<<std::endl;
					for (int k = 0; k < nvars; k++)
					{
						idx[k] = nvars*j+k;
					}
					ierr = VecGetValues(rvec,nvars,idx,relem);CHKERRQ(ierr);
					ierr = VecGetValues(rvec_L,nvars,idx,relem_pert);CHKERRQ(ierr);

					for (int k = 0; k < nvars; k++)
					{
						val = (relem_pert[k]-relem[k])/pertmag;
						ierr = VecSetValues(sum,1,&k,&val,ADD_VALUES);CHKERRQ(ierr);
					}
					ierr = VecAssemblyBegin(sum);CHKERRQ(ierr);
					ierr = VecAssemblyEnd(sum);CHKERRQ(ierr);
					
				}
				
				
								
				
				ierr = VecAXPY(tempvec,-1,sum);CHKERRQ(ierr);
				
				ierr = MatMult(Dinv_i, tempvec, sum);CHKERRQ(ierr); //temporarily store the product in sum

				// ######### Update z_i ############
				for (int k = 0; k < nvars; k++)
				{
					ierr = VecGetValues(sum,1,&k,&val);CHKERRQ(ierr);
					idx[k] = nvars*i+k;
					ierr = VecSetValues(z,1,&idx[k],&val,INSERT_VALUES);CHKERRQ(ierr);
					ierr = VecSetValues(zelem,1,&k,&val,INSERT_VALUES);CHKERRQ(ierr);
				}
				ierr = VecAssemblyBegin(z);CHKERRQ(ierr);
				ierr = VecAssemblyEnd(z);CHKERRQ(ierr);

				ierr = VecAssemblyBegin(zelem);CHKERRQ(ierr);
				ierr = VecAssemblyEnd(zelem);CHKERRQ(ierr);


				// ########## Get Redisuals for doing r(u+pertmag*y) for U-Loop ##########
				pertmag = shell->epsilon_calc(shell->uvec, y);
				Vec uvec_Upert,rvec_U;
				ierr = VecDuplicate(shell->uvec,&rvec_U);CHKERRQ(ierr);
				ierr = VecDuplicate(shell->uvec,&uvec_Upert);CHKERRQ(ierr);
				ierr = VecWAXPY(uvec_Upert,pertmag,y,shell->uvec);CHKERRQ(ierr);
				shell->space->compute_residual(uvec_Upert, rvec_U, false, NULL); CHKERRQ(ierr);
				ierr = VecGhostUpdateBegin(rvec_U, ADD_VALUES, SCATTER_REVERSE); CHKERRQ(ierr);
				ierr = VecGhostUpdateEnd(rvec_U, ADD_VALUES, SCATTER_REVERSE); CHKERRQ(ierr);

				
				
				
				// #################    U-Loop	#####################
				ierr = VecSet(sum,0);CHKERRQ(ierr); // Initialize sum to 0
				for (int j = i+1; j < nelem; j++)
				{//std::cout<<"i,j-------"<<i<<"  "<<j<<std::endl;
					for (int k = 0; k < nvars; k++)
					{
						idx[k] = nvars*j+k;
					}
					ierr = VecGetValues(rvec,nvars,idx,relem);CHKERRQ(ierr);
					ierr = VecGetValues(rvec_U,nvars,idx,relem_pert);CHKERRQ(ierr);

					for (int k = 0; k < nvars; k++)
					{
						val = (relem_pert[k]-relem[k])/pertmag;
						ierr = VecSetValues(sum,1,&k,&val,ADD_VALUES);CHKERRQ(ierr);
					}
					ierr = VecAssemblyBegin(sum);CHKERRQ(ierr);
					ierr = VecAssemblyEnd(sum);CHKERRQ(ierr);
				}

				ierr = MatMult(Dinv_i, sum, tempvec);CHKERRQ(ierr); 
				ierr = VecWAXPY(sum,-1,tempvec,zelem); CHKERRQ(ierr); // sum = z_i - Dinv_i*sum
				
				PetscReal summax,zelemmax;
				ierr = VecMax(sum,NULL,&summax);CHKERRQ(ierr);
				ierr = VecMax(zelem,NULL,&zelemmax);CHKERRQ(ierr);
				if (PetscIsNanReal(summax) || PetscIsNanReal(zelemmax))
				{
					writePetscObj(sum,"yelem");
					writePetscObj(zelem,"zelem");
					std::cout<< "i......."<<i<<std::endl;
					writePetscObj(y,"y");
					writePetscObj(z,"z");
					return -1;
				}

				for (int k = 0; k < nvars; k++)
				{
					ierr = VecGetValues(sum,1,&k,&val);CHKERRQ(ierr);
					idx[k] = nvars*i+k;
					ierr = VecSetValues(y,1,&idx[k],&val,INSERT_VALUES);CHKERRQ(ierr);
				}
				ierr = VecAssemblyBegin(y);CHKERRQ(ierr);
				ierr = VecAssemblyEnd(y);CHKERRQ(ierr);
				
				
			}
			writePetscObj(y,"y");
			writePetscObj(z,"z");
			return -1;
			ierr = VecWAXPY(diff,-1.0,y,yold);CHKERRQ(ierr);
			ierr = VecNorm(diff,NORM_2,&nrm);CHKERRQ(ierr);
			std::cout<<nrm<<"nrm"<<std::endl;
			iter = iter+1;
			std::cout<<iter<<"iter"<<std::endl;
		}
		
		return 0;
		
	
	}
	template
	PetscErrorCode mf_pc_apply2<NVARS>(PC pc, const Vec x, Vec y);
	template
	PetscErrorCode mf_pc_apply2<1>(PC pc, const Vec x, Vec y);

	template<int nvars>
	PetscErrorCode mf_pc_apply3(PC pc, Vec x, Vec y){

		//Checking whether M*randomvec is the same when applied using matrices and matrix free. 

		int ierr = 0;
		MatrixFreePreconditioner<nvars> *shell;
		ierr = PCShellGetContext(pc,&shell);CHKERRQ(ierr);
		int nelem = (shell->n)/nvars;

		//(D+L)D^{-1}(D+U) applies using matrices
		Vec v, tempvec, matdepres, diff, matfreeres; 
		ierr = VecDuplicate(shell->uvec,&v);CHKERRQ(ierr);
		ierr = VecDuplicate(shell->uvec,&tempvec);CHKERRQ(ierr); 
		ierr = VecDuplicate(shell->uvec,&matdepres);CHKERRQ(ierr);
		ierr = VecDuplicate(shell->uvec,&matfreeres);CHKERRQ(ierr);
		ierr = VecDuplicate(shell->uvec,&diff);CHKERRQ(ierr); // difference between matdepres and matfreeres
		init_rand(v); // random vector v

		PetscReal nrm;
		ierr = VecNorm(v,NORM_2,&nrm);CHKERRQ(ierr);
		std::cout<<"v nrm------"<<nrm<<std::endl;

		Mat A, tempmat;
		ierr= PCGetOperators(pc,NULL,&A);CHKERRQ(ierr);
		ierr = MatDuplicate(A,MAT_DO_NOT_COPY_VALUES,&(tempmat)); CHKERRQ(ierr);
		ierr = MatAXPY(shell->Umat,1,shell->D, SAME_NONZERO_PATTERN); CHKERRQ(ierr); //(D+U)
		ierr = MatAXPY(shell->Lmat,1,shell->D, SAME_NONZERO_PATTERN); CHKERRQ(ierr); //(D+L)

		//matdepres = (D+U)*v
		ierr = MatMult(shell->Umat,v,matdepres); CHKERRQ(ierr);

		//tempvec = D^{-1}*matdepres
		ierr = MatMult(shell->Dinv,matdepres,tempvec); CHKERRQ(ierr);

		//matdepres = (D+L)*tempvec
		ierr = MatMult(shell->Lmat,tempvec,matdepres); CHKERRQ(ierr);


		//Initiate the mat-free computation of the above result. 

		//matfreeres = (D+U)*v
		ierr = VecSet(matfreeres,0); CHKERRQ(ierr);// Initializing matfreeres to 0. Just to be sure
		Vec rpert;
		ierr = VecDuplicate(shell->uvec,&rpert);CHKERRQ(ierr);
		PetscScalar pertmag = shell->epsilon_calc(shell->uvec, v);
		pertmag = 1;
		ierr = VecWAXPY(tempvec,pertmag,v,shell->uvec);CHKERRQ(ierr);
		shell->space->compute_residual(tempvec, rpert, false, NULL); CHKERRQ(ierr);
		ierr = VecGhostUpdateBegin(rpert, ADD_VALUES, SCATTER_REVERSE); CHKERRQ(ierr);
		ierr = VecGhostUpdateEnd(rpert, ADD_VALUES, SCATTER_REVERSE); CHKERRQ(ierr);

		for (int i = 0; i < nelem; i++)
		{
			
			PetscInt idx[nvars];
			PetscScalar relem[nvars], relem_pert[nvars], sum[nvars], val;

			for (int k = 0; k < nvars; k++)
			{
				sum[k] = 0.0;
			}
			
			for (int j = i; j < nelem; j++)
			{
				for (int k = 0; k < nvars; k++)
				{
					idx[k] = nvars*j+k;
				}
				
				ierr = VecGetValues(shell->rvec,nvars,idx,relem);CHKERRQ(ierr);
				ierr = VecGetValues(rpert,nvars,idx,relem_pert);CHKERRQ(ierr);

				for (int k = 0; k < nvars; k++)
				{
					val = (relem_pert[k]-relem[k])/pertmag;
					sum[k] = sum[k]+val;
				}
				
			}


			for (int k = 0; k < nvars; k++)
			{
				idx[k] = nvars*i+k;
			}
			ierr = VecSetValues(matfreeres,nvars, idx,sum,INSERT_VALUES);CHKERRQ(ierr); 

			
		}
		ierr = VecAssemblyBegin(matfreeres);CHKERRQ(ierr);
		ierr = VecAssemblyEnd(matfreeres);CHKERRQ(ierr);

		//tempvec = D_inv*matfreeres
		ierr = MatMult(shell->Dinv,matfreeres,tempvec); CHKERRQ(ierr);

		//matfreeres = (D+L)*tempvec

		pertmag = shell->epsilon_calc(shell->uvec, tempvec);
		pertmag = 1;
		ierr = VecWAXPY(matfreeres,pertmag,tempvec,shell->uvec);CHKERRQ(ierr);
		shell->space->compute_residual(matfreeres, rpert, false, NULL); CHKERRQ(ierr);
		ierr = VecGhostUpdateBegin(rpert, ADD_VALUES, SCATTER_REVERSE); CHKERRQ(ierr);
		ierr = VecGhostUpdateEnd(rpert, ADD_VALUES, SCATTER_REVERSE); CHKERRQ(ierr);


		for (int i = 0; i < nelem; i++)
		{
			
			PetscInt idx[nvars];
			PetscScalar relem[nvars], relem_pert[nvars], sum[nvars], val;

			for (int k = 0; k < nvars; k++)
			{
				sum[k] = 0.0;
			}
			
			for (int j = 0; j <= i; j++)
			{
				for (int k = 0; k < nvars; k++)
				{
					idx[k] = nvars*j+k;
				}
				
				ierr = VecGetValues(shell->rvec,nvars,idx,relem);CHKERRQ(ierr);
				ierr = VecGetValues(rpert,nvars,idx,relem_pert);CHKERRQ(ierr);

				for (int k = 0; k < nvars; k++)
				{
					val = (relem_pert[k]-relem[k])/pertmag;
					sum[k] = sum[k]+val;
				}
				
			}


			for (int k = 0; k < nvars; k++)
			{
				idx[k] = nvars*i+k;
			}
			ierr = VecSetValues(matfreeres,nvars, idx,sum,INSERT_VALUES);CHKERRQ(ierr); 

			
		}
		ierr = VecAssemblyBegin(matfreeres);CHKERRQ(ierr);
		ierr = VecAssemblyEnd(matfreeres);CHKERRQ(ierr);

		ierr = VecWAXPY(diff,-1.0,matdepres,matfreeres);CHKERRQ(ierr);
		ierr = VecNorm(diff,NORM_2,&nrm);CHKERRQ(ierr);
		
		std::cout<<"nrm------"<<nrm<<std::endl;
		return -1;
		return 0;

	}

	template
	PetscErrorCode mf_pc_apply3<NVARS>(PC pc, const Vec x, Vec y);
	template
	PetscErrorCode mf_pc_apply3<1>(PC pc, const Vec x, Vec y);


	//template <int nvars>
	//StatusCode MatrixFreePreconditioner::
	template<int nvars>
	PetscErrorCode mf_pc_destroy(PC pc)
	{
	// Set up matrix free PC
	StatusCode ierr = 0;
	// Vec diag;

	MatrixFreePreconditioner<nvars> *shell;
	ierr = PCShellGetContext(pc,&shell);CHKERRQ(ierr);
	ierr = MatDestroy(&shell->Lmat);CHKERRQ(ierr);
	ierr = MatDestroy(&shell->Umat);CHKERRQ(ierr);
	ierr = MatDestroy(&shell->Dinv);CHKERRQ(ierr);
	ierr = PetscFree(shell);CHKERRQ(ierr);
	return 0;

	}

	template
	PetscErrorCode mf_pc_destroy<NVARS>(PC pc);
	template
	PetscErrorCode mf_pc_destroy<1>(PC pc);

	PetscErrorCode init_rand(Vec &v)
	{
		PetscRandom   rctx ;
		PetscRandomCreate(PETSC_COMM_WORLD,&rctx);
		PetscRandomSetSeed(rctx,3); // set seed. const seed ensures same random numbers are generated in diff machines
		PetscRandomSeed(rctx);
		VecSetRandom(v,rctx);
		PetscRandomDestroy(&rctx);
		return 0;

	}

PetscErrorCode writePetscObj(Mat &A, std::string name)
	
	{

		PetscViewer viewer;

		

		const std::string namefin = name + ".m";
		PetscCall(MatView(A, PETSC_VIEWER_STDOUT_WORLD));
		PetscCall(PetscPrintf(PETSC_COMM_WORLD, "writing matrix ...\n"));
		PetscCall(PetscViewerASCIIOpen(PETSC_COMM_WORLD, namefin.c_str(), &viewer));
		PetscCall(PetscViewerPushFormat(viewer, PETSC_VIEWER_ASCII_MATLAB));
        PetscCall(MatView(A, viewer));

		
		// PetscCall(PetscViewerASCIIOpen(PETSC_COMM_WORLD, namefin.c_str(), &viewer));
		// PetscCall(MatView(v, viewer));
		PetscCall(PetscViewerDestroy(&viewer));
		return 0;

	} 

	PetscErrorCode writePetscObj(Vec &v, std::string name)
	
	{

		PetscViewer viewer;
		const std::string namefin = name + ".dat";
		PetscCall(VecView(v, PETSC_VIEWER_STDOUT_WORLD));

		PetscCall(PetscPrintf(PETSC_COMM_WORLD, "writing vector ...\n"));
		PetscCall(PetscViewerASCIIOpen(PETSC_COMM_WORLD, namefin.c_str(), &viewer));
		PetscCall(VecView(v, viewer));
		PetscCall(PetscViewerDestroy(&viewer));
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



