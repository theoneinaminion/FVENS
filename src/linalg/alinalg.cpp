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
		
		//double pertmag = (1e-6)/nrm2;
		return eps;

}




	//template <int nvars>
	//StatusCode MatrixFreePreconditioner::
	template<int nvars, typename scalar>
	PetscErrorCode mf_pc_create(MatrixFreePreconditioner<nvars> **shell){
	// Set up matrix free PC
	StatusCode ierr = 0;
	MatrixFreePreconditioner<nvars> *newctx;
	ierr = PetscNew(&newctx);CHKERRQ(ierr);

	*shell = newctx;
	return 0;
	}
	
	template
	PetscErrorCode mf_pc_create<NVARS,freal>(MatrixFreePreconditioner<NVARS> **shell);
	template
	PetscErrorCode mf_pc_create<1,freal>(MatrixFreePreconditioner<1> **shell);

	// template <int nvars>
	// StatusCode MatrixFreePreconditioner::
	template<int nvars, typename scalar>
	PetscErrorCode mf_pc_setup(PC pc, Vec u, Vec r,const Spatial<freal,nvars> *const space, MatrixFreePreconditioner<nvars> *shell, const Vec dtmvec){
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

	ierr = VecDuplicate(dtmvec,&(shell->mdtvec));CHKERRQ(ierr);
	ierr = VecCopy(dtmvec,shell->mdtvec);CHKERRQ(ierr); // Time step vector

	ierr = MatDuplicate(A,MAT_DO_NOT_COPY_VALUES,&(shell->Dinv)); CHKERRQ(ierr);
	ierr = MatGetBlockSize(A,&(shell->blk_size));CHKERRQ(ierr);
	ierr = MatGetSize(A, &(shell->m), &(shell->n));CHKERRQ(ierr); // get matrix size  
	shell->space = space; 

	ierr = MatScale(shell->Dinv,0); CHKERRQ(ierr);
	ierr = MatInvertBlockDiagonalMat(A,shell->Dinv); CHKERRQ(ierr);
	
	//shell->getLU(A);
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
	PetscErrorCode mf_pc_setup<NVARS,freal>(PC pc, Vec u, Vec r,const Spatial<freal,NVARS> *const space, MatrixFreePreconditioner<NVARS> *shell, const Vec dtmvec);
	template
	PetscErrorCode mf_pc_setup<1,freal>(PC pc, Vec u, Vec r,const Spatial<freal,1> *const space, MatrixFreePreconditioner<1> *shell, const Vec dtmvec);
	

	//template <int nvars>
	//StatusCode MatrixFreePreconditioner::
	template<int nvars, typename scalar>
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
	PetscErrorCode mf_pc_apply<NVARS,freal>(PC pc, Vec x, Vec y);
	template
	PetscErrorCode mf_pc_apply<1,freal>(PC pc, Vec x, Vec y);



	
	# if 1
	template<int nvars, typename scalar>
	PetscErrorCode mf_pc_apply1(PC pc,const Vec x, Vec y)
	{
		//Matrix-dependant LU-SGS preconditioner
		Vec z,temp1,y1, temp2;
		PetscErrorCode ierr;

		MatrixFreePreconditioner<nvars> *shell;
		ierr = PCShellGetContext(pc,&shell);CHKERRQ(ierr);

		//PetscScalar epsilon = 1e-6;
		ierr = VecDuplicate(shell->uvec,&z);CHKERRQ(ierr); 
		ierr = VecDuplicate(shell->uvec,&temp1);CHKERRQ(ierr);
		ierr = VecDuplicate(shell->uvec,&y1);CHKERRQ(ierr);
		ierr = VecDuplicate(shell->uvec,&temp2);CHKERRQ(ierr);

		ierr = VecCopy(x,z);CHKERRQ(ierr);
		ierr = VecCopy(x,y1);CHKERRQ(ierr);


		PetscScalar nrm=10.0;
		PetscInt it=0;
		PetscScalar tol=1e-3;
		while ((nrm > tol)&&(it <= 300))
		{	
			it = it+1;
			//Adding the time term
			// const UMesh<freal,NDIM> *const m = shell->space->mesh();
			// for(fint iel = 0; iel < m->gnelemglobal(); iel++)
			// {
			// 	PetscScalar dt;
			// 	ierr = VecGetValues(shell->mdtvec,1,&iel,&dt);CHKERRQ(ierr);
			// 	for(int i = 0; i < nvars; i++) 
			// 	{
			// 		PetscScalar val;
			// 		PetscInt idx = iel*nvars+i;
			// 		ierr = VecGetValues(y1,1,&idx,&val);CHKERRQ(ierr);
			// 		val = 0*dt;
			// 		ierr = VecSetValues(temp1,1,&idx,&val,INSERT_VALUES);CHKERRQ(ierr);// Adding the time term to r_pert
			// 	}
			// }
			//#### DO NOT ADD TIME TERM ##############
			ierr = MatMult(shell->Lmat,z,temp1);CHKERRQ(ierr);
			// writePetscObj(temp1,"m_l");
		 	//  return -1;
			//ierr = VecAXPY(temp1,1.0,temp2);CHKERRQ(ierr); //temp1 = temp1 + temp2 = (vol/dt * y + L*z)
			ierr = VecAYPX(temp1,-1.0,x);CHKERRQ(ierr); //temp1 = x - temp1 = (x - (vol/dt * y + L*z))

			// writePetscObj(temp1,"x-m_L");
		 	//  return -1;

			ierr = MatMult(shell->Dinv,temp1,temp2);CHKERRQ(ierr); //temp2 = Dinv(temp1) = Dinv(x - (vol/dt * y + L*z))
			// writePetscObj(temp2,"Dinv_x-m_L");
		 	//  return -1;
			ierr = VecCopy(temp2,z);CHKERRQ(ierr); //z = temp2

			//U loop
			ierr = MatMult(shell->Umat,y1,temp1);CHKERRQ(ierr);
			// writePetscObj(temp1,"m_u");
			// return -1;
			ierr = MatMult(shell->Dinv,temp1,temp2);CHKERRQ(ierr); //temp2 = Dinv(U*y1)

			// writePetscObj(temp2,"Dinv_m_u");
		 	// return -1;

			ierr = VecAYPX(temp2,-1.0,z); //temp2 = z - temp2 = z - Dinv(U*y1) 
			// writePetscObj(temp2,"z-Dinv_m_u");
			// return -1;
			ierr = VecWAXPY(temp1,-1.0,temp2,y1); //y^{k+1}-y^{k}
			ierr = VecNorm(temp1,NORM_2,&nrm);CHKERRQ(ierr); //||y^{k+1}-y^{k}||_2
			ierr = VecCopy(temp2,y1); //y^{k+1} = y^{k+1}-y^{k}

			//std::cout<<nrm<<std::endl;
			//  if (it==2)
			// {	writePetscObj(y1,"y12mat");
			// 	writePetscObj(z,"z2mat");
			// 	return -1;
			// }
			// return -1;
		}
		std::cout<<nrm<<std::endl;
		//std::cout<<nrm<<std::endl;
		ierr = VecCopy(y1,y);CHKERRQ(ierr);
		// writePetscObj(y1,"y1");
		// return -1;
		return ierr;
	}
	

	template
	PetscErrorCode mf_pc_apply1<NVARS,freal>(PC pc, const Vec x, Vec y);
	template
	PetscErrorCode mf_pc_apply1<1,freal>(PC pc, const Vec x, Vec y);
	# endif

	template<int nvars, typename scalar>
	PetscErrorCode mf_pc_apply2(PC pc, Vec x, Vec y)
	{

		
		/**
		 * @brief Main scheme is: (D+L)Dinv(D+U)y = x; 
		 * y is the preconditioned vector.
		 * x is the non-preconditioned vector.
		 * 
		 * Matrix-vector products are applied as follows:
		 * (L)v = vol/dt * v + (R_L(u+\epsilon/||v||_2 * v) - R_L(u))/(\epsilon/||v||_2)
		 * (U)v = vol/dt * v + (R_R(u+\epsilon/||v||_2 * v) - R_R(u))/(\epsilon/||v||_2)
		 *
		 * where v is some vector. 
		 * R_L is the residual calculated by equating the fluxes from right side elem = 0
		 * R_R is the residual calculated by equating the fluxes from left side elem = 0
		 */
		
		StatusCode ierr = 0;
		MatrixFreePreconditioner<nvars> *shell;
		ierr = PCShellGetContext(pc,&shell);CHKERRQ(ierr);

		//PetscScalar epsilon = 1e-6;

		Vec z;
		ierr = VecDuplicate(shell->uvec,&z);CHKERRQ(ierr); 


		/**
		 * @brief Left sweep is done first.
		 * (D+L)z = x; 	Dinv(D+U)y = z. 
		 * Here, we will take care of solving (D+L)z = x
		 * 
		 * An iterative can be written as: 
		 * z^{k+1} = Dinv(x-Lz^{k})
		 *  
		 */ 
		
		Vec r_orig, r_orig2, r_pert, u_pert, temp, matfreeprod, matfreeprodU;
		ierr = VecDuplicate(shell->uvec,&r_orig);CHKERRQ(ierr);
		ierr = VecDuplicate(shell->uvec,&r_orig2);CHKERRQ(ierr);
		ierr = VecDuplicate(shell->uvec,&r_pert);CHKERRQ(ierr);
		ierr = VecDuplicate(shell->uvec,&u_pert);CHKERRQ(ierr);
		ierr = VecDuplicate(shell->uvec,&matfreeprod);CHKERRQ(ierr); // the time term.
		ierr = VecDuplicate(shell->uvec,&matfreeprodU);CHKERRQ(ierr);
		ierr = VecDuplicate(shell->uvec,&temp);CHKERRQ(ierr);

		ierr = VecCopy(x,z);CHKERRQ(ierr);// Initial guess for z. NEVER SET TO 0.
		ierr = VecCopy(x,y);CHKERRQ(ierr);// Initial guess for y. NEVER SET TO 0.



		{
			const UMesh<freal,NDIM> *const m = shell->space->mesh();
			MutableVecHandler<PetscScalar> ygh(r_orig);
			PetscScalar *const ygr = ygh.getArray(); 

			MutableVecHandler<PetscScalar> ygk(r_orig2);
			PetscScalar *const ygs = ygk.getArray();

	#pragma omp parallel for simd default(shared)
			for(fint i = 0; i < m->gnelem()*nvars; i++) {
				ygr[i] = 0;ygs[i] = 0;
			}

	#pragma omp parallel for simd default(shared)
			for(fint i = m->gnelem(); i < m->gnelem()+m->gnConnFace(); i++) {
				ygr[i] = 0;ygs[i] = 0;
			}
		}

		//ierr = shell->space->compute_residual(shell->uvec, r_orig, false, NULL); CHKERRQ(ierr); //Residual from left elements only.
		ierr = shell->space->compute_residual_LU(shell->uvec, r_orig, 1); CHKERRQ(ierr); //Residual from left elements only.
		ierr = VecGhostUpdateBegin(r_orig, ADD_VALUES, SCATTER_REVERSE); CHKERRQ(ierr);
		ierr = VecGhostUpdateEnd(r_orig, ADD_VALUES, SCATTER_REVERSE); CHKERRQ(ierr);

		ierr = shell->space->compute_residual_LU(shell->uvec, r_orig2, 2); CHKERRQ(ierr); //Residual from left elements only.
		ierr = VecGhostUpdateBegin(r_orig2, ADD_VALUES, SCATTER_REVERSE); CHKERRQ(ierr);
		ierr = VecGhostUpdateEnd(r_orig2, ADD_VALUES, SCATTER_REVERSE); CHKERRQ(ierr);

		const PetscScalar tol = 1e-3; //Left sweep solved until the ||z^{k+1}-z^{k}||_2 <= 1e-3.
		PetscScalar nrm = 10.; //Initial value of the norm.

		int it = 0;

		while ((nrm > tol)&&(it <= 300))
		{	
			it = it+1;
			PetscScalar znrm; 
			ierr = VecNorm(z,NORM_2,&znrm);CHKERRQ(ierr);
			PetscScalar pertmag = shell->epsilon_calc(shell->uvec,z)/znrm;//epsilon;///znrm;
			//PetscScalar fac = 1000.0*pertmag;
			//std::cout<<pertmag<<"pertmag"<<std::endl;

			// if(znrm < 10.0*std::numeric_limits<freal>::epsilon())
			// 		SETERRQ(PETSC_COMM_SELF, PETSC_ERR_FP,
			// 		"Norm of offset is too small for finite difference Matrix-Vector product!");

			{
				const UMesh<freal,NDIM> *const m = shell->space->mesh();
				ConstVecHandler<PetscScalar> uh(shell->uvec);
				const PetscScalar *const ur = uh.getArray();

				ConstVecHandler<PetscScalar> zh(z);
				const PetscScalar *const zr = zh.getArray();

				MutableVecHandler<PetscScalar> ygh(r_pert);
				PetscScalar *const ygr = ygh.getArray();

				//MutableVecHandler<PetscScalar> ygk(matfreeprod);
				//PetscScalar *const ygs = ygk.getArray(); 

				MutableVecHandler<PetscScalar> auxh(u_pert);
				PetscScalar *const auxr = auxh.getArray();

		#pragma omp parallel for simd default(shared)
				for(fint i = 0; i < m->gnelem()*nvars; i++) {
					ygr[i] = 0; //ygs[i] = 0;
					auxr[i] = ur[i] + pertmag * zr[i];
				}

		#pragma omp parallel for simd default(shared)
				for(fint i = m->gnelem(); i < m->gnelem()+m->gnConnFace(); i++) {
					ygr[i] = 0; //ygs[i] = 0;
				}
			}

			ierr = VecGhostUpdateBegin(u_pert, INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);
			ierr = VecGhostUpdateEnd(u_pert, INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);
			
			//ierr = VecWAXPY(u_pert,pertmag,z,shell->uvec);CHKERRQ(ierr); // u_pert =  shell->uvec + pertmag*z 


			//ierr = shell->space->compute_residual(u_pert, r_pert, false, NULL); CHKERRQ(ierr); //Residual from left elements only with state vec u_pert.
			ierr = shell->space->compute_residual_LU(u_pert, r_pert, 1); CHKERRQ(ierr); //Residual from left elements only with state vec u_pert.
			ierr = VecGhostUpdateBegin(r_pert, ADD_VALUES, SCATTER_REVERSE); CHKERRQ(ierr);
			ierr = VecGhostUpdateEnd(r_pert, ADD_VALUES, SCATTER_REVERSE); CHKERRQ(ierr);
		#if 1
			{
				const UMesh<freal,NDIM> *const m = shell->space->mesh();

				ConstVecHandler<PetscScalar> yh(y);
				const PetscScalar *const yr = yh.getArray();

				ConstVecHandler<PetscScalar> resorig(r_orig);
				const PetscScalar *const reso = resorig.getArray();

				ConstVecHandler<PetscScalar> respert(r_pert);
				const PetscScalar *const resp = respert.getArray();

				ConstVecHandler<PetscScalar> xh(x);
				const PetscScalar *const xr = xh.getArray();

				ConstVecHandler<PetscScalar> mdth(shell->mdtvec);
				const PetscScalar *const dtmr = mdth.getArray();

				MutableVecHandler<PetscScalar> th(matfreeprod);
				PetscScalar *const mfp = th.getArray();

		#pragma omp parallel for simd default(shared)
				for(fint iel = 0; iel < m->gnelem(); iel++)
				{
					for(int i = 0; i < nvars; i++) {
						// finally, add the pseudo-time term (Vol/dt du = Vol/dt x)
						mfp[iel*nvars+i] = xr[iel*nvars+i] - (0*dtmr[iel]*yr[iel*nvars+i]
							+ (resp[iel*nvars+i] - reso[iel*nvars+i])/pertmag); // x - Lz
					}
				}
			}

			ierr = VecGhostUpdateBegin(matfreeprod, ADD_VALUES, SCATTER_REVERSE); CHKERRQ(ierr);
			ierr = VecGhostUpdateEnd(matfreeprod, ADD_VALUES, SCATTER_REVERSE); CHKERRQ(ierr);
		#endif 	

			// temp = Dinv(matfreeprod) = Dinv(x-Lz)
			ierr = MatMult(shell->Dinv, matfreeprod, temp);CHKERRQ(ierr); // temp = Dinv(r_pert)

			// z^{k+1} = temp . So, calculating the norm for this iteration
			//ierr = VecAXPY(z,1.0,temp);CHKERRQ(ierr); // z = z + temp

			//ierr = VecNorm(temp,NORM_2,&nrm);CHKERRQ(ierr); // nrm =  z^{k+1} - z^{k}
			ierr = VecCopy(temp,z);CHKERRQ(ierr); //z^{k+1} = temp
			
			
			//Now solving the second equation.
			//#####################################################################
			PetscScalar ynrm;
			ierr = VecNorm(y,NORM_2,&ynrm);CHKERRQ(ierr);
			pertmag = shell->epsilon_calc(shell->uvec,y)/ynrm;
			{
				const UMesh<freal,NDIM> *const m = shell->space->mesh();
				ConstVecHandler<PetscScalar> uh(shell->uvec);
				const PetscScalar *const ur = uh.getArray();

				ConstVecHandler<PetscScalar> yh(y);
				const PetscScalar *const yr = yh.getArray();

				MutableVecHandler<PetscScalar> ygh(r_pert);
				PetscScalar *const ygr = ygh.getArray();

				MutableVecHandler<PetscScalar> ygk(matfreeprod);
				PetscScalar *const ygs = ygk.getArray(); 

				MutableVecHandler<PetscScalar> auxh(u_pert);
				PetscScalar *const auxr = auxh.getArray();

		#pragma omp parallel for simd default(shared)
				for(fint i = 0; i < m->gnelem()*nvars; i++) {
					ygr[i] = 0; ygs[i] = 0;
					auxr[i] = ur[i] + pertmag * yr[i];
				}

		#pragma omp parallel for simd default(shared)
				for(fint i = m->gnelem(); i < m->gnelem()+m->gnConnFace(); i++) {
					ygr[i] = 0; ygs[i] = 0;
				}
			}

			ierr = VecGhostUpdateBegin(u_pert, INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);
			ierr = VecGhostUpdateEnd(u_pert, INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);

			ierr = shell->space->compute_residual_LU(u_pert, r_pert, 2); CHKERRQ(ierr); //Residual from left elements only with state vec u_pert.
			ierr = VecGhostUpdateBegin(r_pert, ADD_VALUES, SCATTER_REVERSE); CHKERRQ(ierr);
			ierr = VecGhostUpdateEnd(r_pert, ADD_VALUES, SCATTER_REVERSE); CHKERRQ(ierr);

			{
				const UMesh<freal,NDIM> *const m = shell->space->mesh();

				ConstVecHandler<PetscScalar> resorig(r_orig2);
				const PetscScalar *const reso = resorig.getArray();

				ConstVecHandler<PetscScalar> respert(r_pert);
				const PetscScalar *const resp = respert.getArray();

				MutableVecHandler<PetscScalar> th(matfreeprod);
				PetscScalar *const mfp = th.getArray();

		#pragma omp parallel for simd default(shared)
				for(fint iel = 0; iel < m->gnelem(); iel++)
				{
					for(int i = 0; i < nvars; i++) {
						// finally, -(D+U)y
						mfp[iel*nvars+i] = -(resp[iel*nvars+i] - reso[iel*nvars+i])/pertmag; // x - Lz
					}
				}
			}

			ierr = VecGhostUpdateBegin(matfreeprod, ADD_VALUES, SCATTER_REVERSE); CHKERRQ(ierr);
			ierr = VecGhostUpdateEnd(matfreeprod, ADD_VALUES, SCATTER_REVERSE); CHKERRQ(ierr);

			// temp = Dinv(matfreeprod) = -Dinv(D+U)y
			ierr = MatMult(shell->Dinv, matfreeprod, temp);CHKERRQ(ierr); // temp = Dinv(r_pert)

			//temp = z - Dinv(D+U)y = z - temp
			ierr = VecAYPX(temp,-1.0,z);CHKERRQ(ierr); // temp = z - temp
			ierr = VecAYPX(y,-1.0,temp);CHKERRQ(ierr); // y = y - temp	
			ierr = VecNorm(y,NORM_2,&nrm);CHKERRQ(ierr); // nrm =  y^{k+1} - y^{k}
			//ierr = VecAYPX
			//ierr = VecAXPY(y,1.0,temp);CHKERRQ(ierr); // y = y + temp
			//ierr = VecNorm(temp,NORM_2,&nrm);CHKERRQ(ierr); // nrm =  y^{k+1} - y^{k}
			ierr = VecCopy(temp,y); CHKERRQ(ierr); //y^{k+1} = temp

			//std::cout<<nrm<<std::endl;
	

		}
		std::cout<<nrm<<std::endl;
		return ierr;
	
	}
	template
	PetscErrorCode mf_pc_apply2<NVARS,freal>(PC pc, const Vec x, Vec y);
	template
	PetscErrorCode mf_pc_apply2<1,freal>(PC pc, const Vec x, Vec y);

	template<int nvars, typename scalar>
	PetscErrorCode mf_pc_apply3(PC pc, Vec x, Vec y)
	{

		//Checking whether M*randomvec is the same when applied using matrices and matrix free. 
		//std::cout<<"mfpcapply3"<<std::endl;
		int ierr = 0;
		MatrixFreePreconditioner<nvars> *shell;

		ierr = PCShellGetContext(pc,&shell);CHKERRQ(ierr);

		//PetscScalar epsilon = 1e-6;

		Vec z;
		ierr = VecDuplicate(shell->uvec,&z);CHKERRQ(ierr); 
		
		Vec r_orig, r_orig2, r_pert, u_pert, temp, matfreeprod, matfreeprodU,y1;
		ierr = VecDuplicate(shell->uvec,&r_orig);CHKERRQ(ierr);
		ierr = VecDuplicate(shell->uvec,&r_orig2);CHKERRQ(ierr);
		ierr = VecDuplicate(shell->uvec,&r_pert);CHKERRQ(ierr);
		ierr = VecDuplicate(shell->uvec,&u_pert);CHKERRQ(ierr);
		ierr = VecDuplicate(shell->uvec,&y1);CHKERRQ(ierr);
		ierr = VecDuplicate(shell->uvec,&matfreeprod);CHKERRQ(ierr); // the time term.
		ierr = VecDuplicate(shell->uvec,&matfreeprodU);CHKERRQ(ierr);
		ierr = VecDuplicate(shell->uvec,&temp);CHKERRQ(ierr);

		ierr = VecCopy(x,z);CHKERRQ(ierr);// Initial guess for z. NEVER SET TO 0.
		ierr = VecCopy(x,y1);CHKERRQ(ierr);// Initial guess for y. NEVER SET TO 0.

		//#######
		ierr = VecSet(r_orig,0);CHKERRQ(ierr);
		ierr = VecSet(r_orig2,0);CHKERRQ(ierr);
	

		//ierr = shell->space->compute_residual(shell->uvec, r_orig, false, NULL); CHKERRQ(ierr); //Residual from left elements only.
		ierr = shell->space->compute_residual_LU(shell->uvec, r_orig, 1); CHKERRQ(ierr); //Residual from left elements only.
		ierr = VecGhostUpdateBegin(r_orig, ADD_VALUES, SCATTER_REVERSE); CHKERRQ(ierr);
		ierr = VecGhostUpdateEnd(r_orig, ADD_VALUES, SCATTER_REVERSE); CHKERRQ(ierr);

		ierr = shell->space->compute_residual_LU(shell->uvec, r_orig2, 2); CHKERRQ(ierr); //Residual from left elements only.
		ierr = VecGhostUpdateBegin(r_orig2, ADD_VALUES, SCATTER_REVERSE); CHKERRQ(ierr);
		ierr = VecGhostUpdateEnd(r_orig2, ADD_VALUES, SCATTER_REVERSE); CHKERRQ(ierr);


		PetscScalar tol = 1e-6;
		PetscScalar nrm = 10.;
		int it = 0;
		int blu = 1;
		while ((nrm>tol) && (it <=300))
		{
			it = it+1;
			PetscScalar znrm;
			ierr = VecNorm(z,NORM_2,&znrm);CHKERRQ(ierr);
			PetscScalar pertmag = shell->epsilon_calc(shell->uvec,z);///znrm;

			ierr = VecSet(u_pert,0);CHKERRQ(ierr);
			ierr = VecSet(r_pert,0);CHKERRQ(ierr);
			ierr = VecSet(matfreeprod,0);CHKERRQ(ierr);

			ierr = VecWAXPY(u_pert,pertmag,z,shell->uvec);CHKERRQ(ierr); // u_pert =  shell->uvec + pertmag*z

			ierr = shell->space->compute_residual_LU(u_pert, r_pert, 1); CHKERRQ(ierr); //Residual from left elements only with state vec u_pert.
			ierr = VecGhostUpdateBegin(r_pert, ADD_VALUES, SCATTER_REVERSE); CHKERRQ(ierr);
			ierr = VecGhostUpdateEnd(r_pert, ADD_VALUES, SCATTER_REVERSE); CHKERRQ(ierr);

			ierr = VecAXPY(r_pert,-1,r_orig);CHKERRQ(ierr); // r_pert = r_pert - r_orig
			ierr = VecScale(r_pert,1./pertmag);CHKERRQ(ierr); // r_pert = r_pert/pertmag = (r_pert - rL)/pertmag

			// writePetscObj(r_pert,"rpert");
		 	// return -1;

			//Adding the time term
			// const UMesh<freal,NDIM> *const m = shell->space->mesh();
			// for(fint iel = 0; iel < m->gnelemglobal(); iel++)
			// {
			// 	PetscScalar dt;
			// 	ierr = VecGetValues(shell->mdtvec,1,&iel,&dt);CHKERRQ(ierr);
			// 	for(int i = 0; i < nvars; i++) 
			// 	{
			// 		PetscScalar val;
			// 		PetscInt idx = iel*nvars+i;
			// 		ierr = VecGetValues(y,1,&idx,&val);CHKERRQ(ierr);
			// 		val = val*dt;
			// 		ierr = VecSetValues(r_pert,1,&idx,&val,ADD_VALUES);CHKERRQ(ierr);// Adding the time term to r_pert
			// 	}
			// }
			// ierr = VecAssemblyBegin(r_pert);CHKERRQ(ierr);
			// ierr = VecAssemblyEnd(r_pert);CHKERRQ(ierr);


			ierr = VecAYPX(r_pert,-1.0,x);CHKERRQ(ierr); // r_pert = x - r_pert

			// writePetscObj(r_pert,"x-mf_l");
			// return -1;

			ierr = MatMult(shell->Dinv, r_pert, temp);CHKERRQ(ierr); // temp = Dinv(r_pert)

			// writePetscObj(temp,"Dinv_x-mf_l");
			// return -1;
			
			ierr = VecCopy(temp,z);CHKERRQ(ierr); //z^{k+1} = temp

			//ierr = VecAXPY(z,1.0,temp);CHKERRQ(ierr); // z = z + temp


			//Now solving the second equation.
			//#####################################################################

			ierr = VecSet(u_pert,0);CHKERRQ(ierr);
			ierr = VecSet(r_pert,0);CHKERRQ(ierr);

			PetscScalar ynrm;
			ierr = VecNorm(y1,NORM_2,&ynrm);CHKERRQ(ierr);
			pertmag = shell->epsilon_calc(shell->uvec,y1)/ynrm;
			

			ierr = VecWAXPY(u_pert,pertmag,y1,shell->uvec);CHKERRQ(ierr); // u_pert =  shell->uvec + pertmag*y

			ierr = shell->space->compute_residual_LU(u_pert, r_pert, 2); CHKERRQ(ierr); //Residual from left elements only with state vec u_pert.
			ierr = VecGhostUpdateBegin(r_pert, ADD_VALUES, SCATTER_REVERSE); CHKERRQ(ierr);
			ierr = VecGhostUpdateEnd(r_pert, ADD_VALUES, SCATTER_REVERSE); CHKERRQ(ierr);

			ierr = VecAXPY(r_pert,-1,r_orig2);CHKERRQ(ierr); // r_pert = r_pert - r_orig
			ierr = VecScale(r_pert,1./pertmag);CHKERRQ(ierr); // r_pert = r_pert/pertmag = (r_pert - rL)/pertmag

			//  writePetscObj(r_pert,"mf_u");
		 	//  return -1;

			ierr = MatMult(shell->Dinv, r_pert, temp);CHKERRQ(ierr); // temp = Dinv(r_pert)
			// writePetscObj(temp,"Dinv_mf_u");
			// return -1;

			if ((it==2) && blu)
			{
				//writePetscObj(temp,"temp");
			 	//std::cout<<"pertmag y"<<pertmag<<std::endl;
			}

			ierr = VecAXPY(temp,-1.0,z);CHKERRQ(ierr); // temp = z - temp
			// writePetscObj(temp,"z-Dinv_mf_u");
			// return -1;
			ierr = VecAXPY(y1,-1.0,temp);CHKERRQ(ierr); // y = y - temp
			ierr = VecNorm(y1,NORM_2,&nrm);CHKERRQ(ierr); // nrm =  y^{k+1} - y^{k}
			ierr = VecCopy(temp,y1); CHKERRQ(ierr); //y^{k+1} = temp
			//ierr = VecAXPY(y,1.0,temp);CHKERRQ(ierr); // y = y + temp
			
		}	
		std::cout<<nrm<<std::endl;
		ierr = VecCopy(y1,y);CHKERRQ(ierr);
		
		 writePetscObj(y1,"ymf");
		 return -1;
		return ierr;

	}

	template
	PetscErrorCode mf_pc_apply3<NVARS, freal>(PC pc, const Vec x, Vec y);
	template
	PetscErrorCode mf_pc_apply3<1, freal>(PC pc, const Vec x, Vec y);





	template<int nvars, typename scalar>
	PetscErrorCode mf_pc_apply4(PC pc, Vec x, Vec y)
	{

		//Checking whether M*randomvec is the same when applied using matrices and matrix free. 
		//std::cout<<"mfpcapply3"<<std::endl;
		int ierr = 0;
		MatrixFreePreconditioner<nvars> *shell;
		ierr = PCShellGetContext(pc,&shell);CHKERRQ(ierr);

			
		MPI_Comm mycomm;
		ierr = PetscObjectGetComm((PetscObject)shell->Dinv, &mycomm); CHKERRQ(ierr);
		const int mpisize = get_mpi_size(mycomm);
		const bool isdistributed = (mpisize > 1);

		using Eigen::Matrix; using Eigen::RowMajor;
		Matrix<freal,nvars,nvars,RowMajor> Dinv; //inverse of the diagonal matrix per element. Data taken from shell->Dinv

		//PetscScalar epsilon = 1e-6;

		Vec z;
		ierr = VecDuplicate(shell->uvec,&z);CHKERRQ(ierr); 
		
		Vec r_orig, r_orig2, r_pertL,r_pertU, u_pert, temp, y1;
		ierr = VecDuplicate(shell->uvec,&r_orig);CHKERRQ(ierr);
		ierr = VecDuplicate(shell->uvec,&r_orig2);CHKERRQ(ierr);
		ierr = VecDuplicate(shell->uvec,&r_pertL);CHKERRQ(ierr);
		ierr = VecDuplicate(shell->uvec,&r_pertU);CHKERRQ(ierr);
		ierr = VecDuplicate(shell->uvec,&u_pert);CHKERRQ(ierr);
		ierr = VecDuplicate(shell->uvec,&y1);CHKERRQ(ierr);
		ierr = VecDuplicate(shell->uvec,&temp);CHKERRQ(ierr);

		ierr = VecCopy(x,z);CHKERRQ(ierr);// Initial guess for z. NEVER SET TO 0.
		ierr = VecCopy(x,y1);CHKERRQ(ierr);// Initial guess for y. NEVER SET TO 0.

		const UMesh<freal,NDIM> *const m = shell->space->mesh();

		PetscScalar tol = 1e-3; 
		PetscScalar nrm = 10.;
		PetscInt it = 0;
		
		
			
		while ((nrm>=tol) && (it <=300))
		{
			it = it+1;
			ierr = VecCopy(y1,temp);CHKERRQ(ierr);// Old value of y1
			PetscScalar znrm;
			ierr = VecNorm(z,NORM_2,&znrm);CHKERRQ(ierr);
			PetscScalar pertmag = shell->epsilon_calc(shell->uvec,z);//znrm;
			pertmag = 1.;
			ierr = VecWAXPY(u_pert,pertmag,z,shell->uvec);CHKERRQ(ierr); // u_pert =  shell->uvec + z
			ierr = shell->space->compute_residual(u_pert, r_pertL, false,NULL); CHKERRQ(ierr); 
			ierr = VecGhostUpdateBegin(r_pertL, ADD_VALUES, SCATTER_REVERSE); CHKERRQ(ierr);
			ierr = VecGhostUpdateEnd(r_pertL, ADD_VALUES, SCATTER_REVERSE); CHKERRQ(ierr);

			//Perturbing with y1
			PetscScalar ynrm;
			ierr = VecNorm(y1,NORM_2,&ynrm);CHKERRQ(ierr);
			pertmag = shell->epsilon_calc(shell->uvec,y1);///ynrm;

			ierr = VecWAXPY(u_pert,pertmag,y1,shell->uvec);CHKERRQ(ierr); // u_pert =  shell->uvec + y1
			ierr = shell->space->compute_residual(u_pert, r_pertU, false,NULL); CHKERRQ(ierr);
			ierr = VecGhostUpdateBegin(r_pertU, ADD_VALUES, SCATTER_REVERSE); CHKERRQ(ierr);
			ierr = VecGhostUpdateEnd(r_pertU, ADD_VALUES, SCATTER_REVERSE); CHKERRQ(ierr);

			ConstVecHandler<scalar> xvh(x);
			const scalar *const xarr = xvh.getArray();
			//Eigen::Map<MVector<const scalar>> x1(xarr, m->gnelem(), NVARS);

			ConstVecHandler<scalar> rvh(shell->rvec);
			const scalar *const rarr = rvh.getArray();
			//Eigen::Map<MVector<const scalar>> res(rarr, m->gnelem(), NVARS);	//original residual

			ConstVecHandler<scalar> rLvh(r_pertL);
			const scalar *const rLarr = rLvh.getArray();
			//Eigen::Map<MVector<const scalar>> resL(rLarr, m->gnelem(), NVARS); //perturbed with z

			ConstVecHandler<scalar> rUvh(r_pertU);
			const scalar *const rUarr = rUvh.getArray();
			//Eigen::Map<MVector<const scalar>> resU(rUarr, m->gnelem(), NVARS); //perturbed with y1

			MutableVecHandler<scalar> zvh(z);
			scalar *const zarr = zvh.getArray();
			//Eigen::Map<MVector<scalar>> zL(zarr, m->gnelem(), NVARS); //z^{k}

			MutableVecHandler<scalar> yvh(y1);
			scalar *const y1arr = yvh.getArray();
			//Eigen::Map<MVector<scalar>> yarr(y1arr, m->gnelem(), NVARS); 


			//Calculate the preconditioned vector element by element..//The below for loop is correct logic. It takes care of the boundary elements/conn as well.
			for(int i=0; i < m->gnelem(); i++)
			{

				int nface = m->gnfael(i); //Number of faces of the element
				PetscInt lidx[nface], ridx[nface]; //Face Ids and Indices of the left and right elements corresponding to the faces

				for(int jface=0; jface<nface; jface++)
				{	
					int face = m->gelemface(i,jface); //Face Id
					int elem = m->gintfac(face,0); //Left element corresponding to the face
					lidx[jface] = (elem!=i) ? elem : -1; //Left element. is equal to -1 if elem is equal to i
					
					elem = m->gintfac(face,1); //Left element corresponding to the face
					ridx[jface] = ((elem!=i)&&(elem<m->gnelem())) ? elem : -1; //Right element. is equal to -1 if elem is equal to i 
					//std::cout<<lidx[jface]<<" "<<ridx[jface]<<std::endl;
					
				}
				
				PetscScalar sum[NVARS]; //To store residual sums in L or U 

				for(int j = 0; j<NVARS; j++)
				{
					sum[j] = 0;
				}


				for (int j = 0; j<nface; j++)
				{
					//Check for L elements
					if (lidx[j] != -1)
					{
						for(int k = 0; k<NVARS; k++)
						{
							sum[k] = sum[k] + (rLarr[lidx[j]*NVARS+k] - rarr[lidx[j]*NVARS+k])/pertmag; // the matrix free product

						}

					}
				}
				PetscInt rows [NVARS]; //To get Dinv for the corresponding element
				const fint element = isdistributed ? m->gglobalElemIndex(i) : i; //Global index number of element in case of parallel run

				Matrix<freal,nvars,1> zcopy,zelem; //This roundabout way of defining a vector somehow works. Using Eigen::Vector gives comple errors
				for (int j = 0; j < NVARS; j++)
				{
					rows[j] = element*NVARS+j;
					zarr[rows[j]] = xarr[rows[j]] - sum[j];
					zcopy[j] = zarr[rows[j]]; //copying to multiply with Dinv

	
				}

				ierr = MatGetValues(shell->Dinv, NVARS, rows, NVARS, rows, Dinv.data());CHKERRQ(ierr);
				
				zelem = Dinv*zcopy; //z^{k+1} = Dinv*(x - sum) = Dinv*(x - Lz


				for (int j = 0; j < NVARS; j++)
				{
					zarr[rows[j]] = zelem[j];
					sum[j] = 0; //Setting the sum back to 0.
				}
				
				//Calculating U sweep
				Matrix<freal,nvars,1> yelem,ycopy;
				for (int j = 0; j<nface; j++)
				{
					//Check for L elements
					if (ridx[j] != -1)
					{
						for(int k = 0; k<NVARS; k++)
						{
							sum[k] = sum[k] + (rUarr[ridx[j]*NVARS+k] - rarr[ridx[j]*NVARS+k])/pertmag; // the matrix free product
							ycopy[k] = sum[k]; //copying to multiply with Dinv
						}
					}
				}
				
				yelem = Dinv*ycopy;// Storing in yelem


				for (int j = 0; j < NVARS; j++)
				{
					y1arr[rows[j]] = zarr[rows[j]] - yelem[j];
				}
				
				//return -1;
			}

			ierr = VecGhostUpdateBegin(z, ADD_VALUES, SCATTER_REVERSE); CHKERRQ(ierr);
			ierr = VecGhostUpdateEnd(z, ADD_VALUES, SCATTER_REVERSE); CHKERRQ(ierr);
			ierr = VecGhostUpdateBegin(y1, ADD_VALUES, SCATTER_REVERSE); CHKERRQ(ierr);
			ierr = VecGhostUpdateEnd(y1, ADD_VALUES, SCATTER_REVERSE); CHKERRQ(ierr);

			ierr = VecAXPY(temp,-1.0,y1);CHKERRQ(ierr); // y1 = y1 - temp
			ierr = VecNorm(temp,NORM_2,&nrm);CHKERRQ(ierr); // nrm =  y1^{k+1} - y1^{k}
			
			std::cout<<nrm<<std::endl;			
		}
		
		ierr = VecCopy(y1,y); CHKERRQ(ierr);
		return ierr;
		
	}

	template
	PetscErrorCode mf_pc_apply4<NVARS, freal>(PC pc, const Vec x, Vec y);
	template
	PetscErrorCode mf_pc_apply4<1, freal>(PC pc, const Vec x, Vec y);






	//template <int nvars>
	//StatusCode MatrixFreePreconditioner::
	template<int nvars, typename scalar>
	PetscErrorCode mf_pc_destroy(PC pc)
	{
	// Set up matrix free PC
	StatusCode ierr = 0;
	// Vec diag;

	MatrixFreePreconditioner<nvars> *shell;
	ierr = PCShellGetContext(pc,&shell);CHKERRQ(ierr);
	//ierr = MatDestroy(&shell->Lmat);CHKERRQ(ierr);
	//ierr = MatDestroy(&shell->Umat);CHKERRQ(ierr);
	ierr = MatDestroy(&shell->Dinv);CHKERRQ(ierr);
	ierr = PetscFree(shell);CHKERRQ(ierr);
	return 0;

	}

	template
	PetscErrorCode mf_pc_destroy<NVARS,freal>(PC pc);
	template
	PetscErrorCode mf_pc_destroy<1,freal>(PC pc);

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
		//PetscCall(MatView(A, PETSC_VIEWER_STDOUT_WORLD));
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
		//PetscCall(VecView(v, PETSC_VIEWER_STDOUT_WORLD));

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



