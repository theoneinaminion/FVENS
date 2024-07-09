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


//################################   SHELL PC ######################################################




template <int nvars,typename scalar>
MatrixFreePreconditiner<nvars,scalar>::MatrixFreePreconditiner(const Spatial<freal,nvars> *const spatial_discretization)
	: space{spatial_discretization}, eps{1e-7}
{
	PetscBool set = PETSC_FALSE;
	PetscOptionsGetReal(NULL, NULL, "-matrix_free_difference_step", &eps, &set);

}


template <int nvars,typename scalar>
void MatrixFreePreconditiner<nvars,scalar>::set_state(const Vec u_state, const Vec r_state)
{
	std::cout<<"In setting state\n";
	u = u_state;
	res = r_state;
}

template <int nvars,typename scalar>
PetscErrorCode MatrixFreePreconditiner<nvars,scalar>::get_LUD(Mat &A)
{
	std::cout<<"In making LUD\n";
	PetscErrorCode ierr = 0;
	//std::cout<<eps<<std::endl;
	ierr = MatDuplicate(A,MAT_COPY_VALUES,&DpL);CHKERRQ(ierr);
	ierr = MatDuplicate(A,MAT_COPY_VALUES,&DpU);CHKERRQ(ierr);
	ierr = MatDuplicate(A,MAT_COPY_VALUES,&D);CHKERRQ(ierr);
	

	PetscInt m;
	PetscInt n;

	MatGetSize(A, &m, &n); // get matrix size 

	int b = m/nvars;
	PetscInt rows[nvars];
	PetscInt cols[nvars];
	PetscScalar Val[nvars*nvars];



	for (int i = 0; i < nvars*nvars; i++)
	{

			Val[i] = 0.0;
		
	}
	
	const PetscScalar *val1 = Val;
	for (int i = 0; i < b; i++)
	{
		//std::cout<<i<<std::endl;
		int p = i*nvars;
		for (int j = 0; j < nvars; j++)
		{
			rows[j] = p+j;
			cols[j] = p+j;
		}

		// zero out the diagonal blocks
		const PetscInt *rows1 = rows;
		//const PetscInt *cols1 = cols;
		// MatSetValues(DpL,nvars,rows1,nvars,cols1,val1,INSERT_VALUES);
		// MatSetValues(DpU,nvars,rows1,nvars,cols1,val1,INSERT_VALUES);
	
		//Choosing to not zero out diagonals
		// Zero out the upper triangular blocks in Lmat and D

		for (int j = i+1; j < b; j++)
		{
			for (int k = 0; k<nvars; k++)
			{
				cols[k] = cols[k] + nvars;

			}
			
			const PetscInt *cols1 = cols;
			ierr = MatSetValues(DpL,nvars,rows1,nvars,cols1,val1,INSERT_VALUES);CHKERRQ(ierr);
			ierr = MatSetValues(D,nvars,rows1,nvars,cols1,val1,INSERT_VALUES);CHKERRQ(ierr);
		}
		
		// zero out the lower triangle blocks in Umat and D
		for (int j = 0; j < nvars; j++)
		{
			
			cols[j] = rows[j];
		}
		for (int j = i-1; j >=0; j--)
		{
			for (int k = 0; k<nvars; k++)
			{
				cols[k] = cols[k] - nvars;

			}
			
			const PetscInt *rows1 = rows;
			const PetscInt *cols1 = cols;
			ierr = MatSetValues(DpU,nvars,rows1,nvars,cols1,val1,INSERT_VALUES);CHKERRQ(ierr);
			ierr = MatSetValues(D,nvars,rows1,nvars,cols1,val1,INSERT_VALUES);CHKERRQ(ierr);
		}

	}

	ierr = MatSetType(DpL,MATAIJ);CHKERRQ(ierr); 
	ierr = MatSetType(DpU,MATAIJ);CHKERRQ(ierr);
	ierr = MatSetType(D,MATAIJ);CHKERRQ(ierr);

	ierr = MatSetUp(DpL);CHKERRQ(ierr);
	ierr = MatSetUp(DpU);CHKERRQ(ierr);
	ierr = MatSetUp(D);CHKERRQ(ierr);
	std::cout<<"Done setting LUD to AIJ\n";

	ierr =MatAssemblyBegin(DpL,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
	MatAssemblyEnd(DpL,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
	MatAssemblyBegin(DpU,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
	MatAssemblyEnd(DpU,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
	MatAssemblyBegin(D,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
	MatAssemblyEnd(D,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
	std::cout<<"Done making LUD\n";

	Mat temp;
	ierr = MatDuplicate(DpL,MAT_COPY_VALUES,&temp);CHKERRQ(ierr);
	//ierr = MatSetup(temp);CHKERRQ(ierr);
	ierr = MatAXPY(temp,1.0,DpU,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
	ierr = MatAXPY(temp,-1.0,D,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
	ierr = MatAXPY(temp,-1.0,A,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
	PetscReal norm;
	ierr = MatNorm(temp,NORM_FROBENIUS,&norm);CHKERRQ(ierr);
	std::cout<<"Norm of L+U+D-A = "<<norm<<std::endl;
	return ierr;

}


template <int nvars,typename scalar>
PetscErrorCode MatrixFreePreconditiner<nvars,scalar>::m_LUSGS(PC pc, Vec x, Vec y)
{
	std::cout<<"In mat LU-SGS apply\n";
	PetscErrorCode ierr = 0;
	MatrixFreePreconditiner<nvars,scalar> *mfpc = nullptr;
	ierr = PCShellGetContext(pc, &mfpc);CHKERRQ(ierr);
	Vec z;
	ierr = VecDuplicate(mfpc->u, &z);CHKERRQ(ierr);

	//Forward sweep
	KSP forward;
	PC forward_pc;

	ierr = KSPCreate(PETSC_COMM_WORLD, &forward);CHKERRQ(ierr);
	ierr = KSPSetType(forward, KSPRICHARDSON);CHKERRQ(ierr);
	ierr = KSPGetPC(forward, &forward_pc);CHKERRQ(ierr);
	ierr = PCSetType(forward_pc, PCSOR);CHKERRQ(ierr);

	ierr = PCSORSetSymmetric(forward_pc, SOR_LOCAL_FORWARD_SWEEP);CHKERRQ(ierr);
	ierr = PCSORSetOmega(forward_pc, 1.0);CHKERRQ(ierr);
	ierr = PCSORSetIterations(forward_pc, 1,1);CHKERRQ(ierr);

	ierr = KSPSetOperators(forward, mfpc->DpL, mfpc->DpL);CHKERRQ(ierr);
	ierr = KSPSolve(forward, x, z);CHKERRQ(ierr);

	//Backward sweep
	KSP backward;
	PC backward_pc;

	Vec temp;
	ierr = VecDuplicate(mfpc->u, &temp);CHKERRQ(ierr);

	ierr = MatMult(mfpc->D, z, temp);CHKERRQ(ierr);

	ierr = KSPCreate(PETSC_COMM_WORLD, &backward);CHKERRQ(ierr);
	ierr = KSPSetType(backward, KSPRICHARDSON);CHKERRQ(ierr);
	ierr = KSPGetPC(backward, &backward_pc);CHKERRQ(ierr);
	ierr = PCSetType(backward_pc, PCSOR);CHKERRQ(ierr);

	ierr = PCSORSetSymmetric(backward_pc, SOR_LOCAL_BACKWARD_SWEEP);CHKERRQ(ierr);
	ierr = PCSORSetOmega(backward_pc, 1.0);CHKERRQ(ierr);

	ierr = PCSORSetIterations(backward_pc, 1,1);CHKERRQ(ierr);

	ierr = KSPSetOperators(backward, mfpc->DpU, mfpc->DpU);CHKERRQ(ierr);
	ierr = KSPSolve(backward, temp, y);CHKERRQ(ierr);

	ierr = VecDestroy(&z);CHKERRQ(ierr);
	ierr = VecDestroy(&temp);CHKERRQ(ierr);
	ierr = KSPDestroy(&forward);CHKERRQ(ierr);
	ierr = KSPDestroy(&backward);CHKERRQ(ierr);
	std::cout<<"Done mat LU-SGS apply\n";
	return ierr;
}


template <int nvars,typename scalar>
PetscErrorCode MatrixFreePreconditiner<nvars,scalar>::testapply(PC pc, Vec x, Vec y) const
{
	PetscErrorCode ierr;
	ierr = MatMult(Dinv, x, y);CHKERRQ(ierr);
	return ierr;
}

template <int nvars,typename scalar>
PetscErrorCode MatrixFreePreconditiner<nvars,scalar>::setup_shell_pc_mlusgs(PC pc)
{
	PetscErrorCode ierr = 0;
	Mat A, Adiag;
	ierr = PCGetOperators(pc, NULL, &A);CHKERRQ(ierr);
	ierr = MatDuplicate(A, MAT_DO_NOT_COPY_VALUES, &Dinv);CHKERRQ(ierr);
	ierr = MatDuplicate(A, MAT_DO_NOT_COPY_VALUES, &Adiag);CHKERRQ(ierr);
	ierr = MatZeroEntries(Dinv);CHKERRQ(ierr);
	ierr = MatZeroEntries(Adiag);CHKERRQ(ierr);
	ierr = MatInvertBlockDiagonalMat(A,Dinv);CHKERRQ(ierr);

	ierr = MatAssemblyBegin(Dinv, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
	ierr = MatAssemblyEnd(Dinv, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
	return -1;
	return ierr;
}

template <int nvars,typename scalar>
PetscErrorCode MatrixFreePreconditiner<nvars,scalar>::setup_shell_pc(PC pc)
{
	PetscErrorCode ierr = 0;
	ierr = setup_shell_pc_mlusgs(pc);CHKERRQ(ierr);
	return ierr;
}


template class MatrixFreePreconditiner<NVARS,freal>;
template class MatrixFreePreconditiner<1,freal>;


template <int nvars,typename scalar>
PetscErrorCode pcapply(PC pc, Vec x, Vec y)
{
	//std::cout<<"In PC Apply\n";
	PetscErrorCode ierr = 0;
	MatrixFreePreconditiner<nvars,scalar> *mfpc = nullptr;
	ierr = PCShellGetContext(pc, &mfpc);CHKERRQ(ierr);

	ierr = mfpc->testapply(pc, x, y); //Select function manually for now
	//std::cout<<"Done PC Apply\n";
	return ierr;
}
template
PetscErrorCode pcapply<NVARS,freal>(PC pc, Vec x, Vec y);
template
PetscErrorCode pcapply<1,freal>(PC pc, Vec x, Vec y);



template <int nvars,typename scalar>
PetscErrorCode pcsetup(PC pc)
{
	//std::cout<<"In PC Setup\n";
	PetscErrorCode ierr = 0;
	MatrixFreePreconditiner<nvars,scalar> *mfpc = nullptr;
	ierr = PCShellGetContext(pc, &mfpc);CHKERRQ(ierr);

	ierr = mfpc->setup_shell_pc(pc);CHKERRQ(ierr);
	//std::cout<<"Done PC Setup\n";
	return ierr;
}
template
PetscErrorCode pcsetup<NVARS,freal>(PC pc);
template
PetscErrorCode pcsetup<1,freal>(PC pc);


template <int nvars,typename scalar>
PetscErrorCode pcdestroy(PC pc)
{
	//std::cout<<"In PC Destroy\n";
	PetscErrorCode ierr = 0;
	MatrixFreePreconditiner<nvars,scalar> *mfpc = nullptr;
	ierr = PCShellGetContext(pc, mfpc);CHKERRQ(ierr);

	ierr = PetscFree(mfpc); CHKERRQ(ierr);		
	return ierr;
}

template
PetscErrorCode pcdestroy<NVARS,freal>(PC pc);
template
PetscErrorCode pcdestroy<1,freal>(PC pc);



template <int nvars,typename scalar>
PetscErrorCode create_shell_precond(const Spatial<freal,nvars> *const spatial, PC *pc)
{
	PetscErrorCode ierr;
	std::cout<<"In Create Shell Precond\n";
	MatrixFreePreconditiner<nvars,scalar> *mfpc = new MatrixFreePreconditiner<nvars,scalar>(spatial);
	//ierr = PetscNew(&mfpc);CHKERRQ(ierr);

	ierr = PCShellSetContext(*pc,mfpc);CHKERRQ(ierr);
	ierr = PCShellSetSetUp(*pc,pcsetup<nvars,scalar>);CHKERRQ(ierr);
	ierr = PCShellSetApply(*pc,pcapply<nvars,scalar>);CHKERRQ(ierr);
	ierr = PCShellSetDestroy(*pc,pcdestroy<nvars,scalar>);CHKERRQ(ierr);

	
	return ierr;
}
template
PetscErrorCode create_shell_precond<NVARS,freal>(const Spatial<freal,NVARS> *const spatial, PC *pc);
template
PetscErrorCode create_shell_precond<1,freal>(const Spatial<freal,1> *const spatial,PC *pc);

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



} // namespace fvens
