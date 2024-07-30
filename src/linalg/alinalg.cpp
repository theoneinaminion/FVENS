#include "alinalg.hpp"
#include "utilities/mpiutils.hpp"
#include <iostream>
#include <vector>
#include <cstring>
#include <limits>
#include <numeric>

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
PetscErrorCode MatrixFreePreconditiner<nvars,scalar>::setup_shell_pc_mlusgs(PC pc)
{
	PetscErrorCode ierr = 0;
	Mat A;
	ierr = PCGetOperators(pc, NULL, &A);CHKERRQ(ierr);
	ierr = MatDuplicate(A, MAT_DO_NOT_COPY_VALUES, &Dinv);CHKERRQ(ierr);
	ierr = MatZeroEntries(Dinv);CHKERRQ(ierr);
	ierr = MatInvertBlockDiagonalMat(A,Dinv);CHKERRQ(ierr);

	ierr = MatAssemblyBegin(Dinv, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
	ierr = MatAssemblyEnd(Dinv, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

	ierr = get_block_LUD(A);CHKERRQ(ierr);
	return ierr;
}

template <int nvars,typename scalar>
PetscErrorCode MatrixFreePreconditiner<nvars,scalar>::get_block_LUD(Mat &A)
{
	PetscErrorCode ierr = 0;
	MPI_Comm mycomm;
	ierr = PetscObjectGetComm((PetscObject)A, &mycomm); CHKERRQ(ierr);
	const int mpisize = get_mpi_size(mycomm);
	const bool isdistributed = (mpisize > 1);

	ierr = MatDuplicate(A, MAT_COPY_VALUES, &DpL);CHKERRQ(ierr); //D+L
	ierr = MatDuplicate(A, MAT_COPY_VALUES, &DpU);CHKERRQ(ierr); //D+U
	ierr = MatDuplicate(A, MAT_DO_NOT_COPY_VALUES, &D);CHKERRQ(ierr); //D initialized to zero

	ierr = MatInvertBlockDiagonalMat(Dinv,D);CHKERRQ(ierr); //it is just easier this way
	MatAssemblyBegin(D, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
	MatAssemblyEnd(D, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);


	const UMesh<freal,NDIM> *const m = space->mesh();

	PetscScalar zeros[nvars*nvars];
	for(int i=0; i<nvars*nvars; i++)
		zeros[i] = 0.0;
	
	//We need to zero out U blocks from DpL and L blocks from DpU
	for(int i=0; i < m->gnelem(); i++)
	{
		const fint element = isdistributed ? m->gglobalElemIndex(i) : i; //Global index number of element in case of parallel run
		int nface = m->gnfael(i); //Number of faces of the element


		for(int jface=0; jface<nface; jface++)
		{
			int nbr_elem = m->gesuel(element,jface); //Neighbour element
			
			if (nbr_elem >=m->gnelem()) 
				continue;

			if(nbr_elem < element) // if mat = 4x4, and i = 3 here, lower triangle elements are all <3. That is the logic.
			{
				const int gnbr_elem = isdistributed ? m->gglobalElemIndex(nbr_elem) : nbr_elem;
				ierr = MatSetValuesBlocked(DpU, 1, &element, 1, &gnbr_elem, zeros, INSERT_VALUES); CHKERRQ(ierr);

			}

			if((nbr_elem > element)) // if mat = 4x4, and i = 3 here, upper triangle elements are all > 3. That is the logic.
			{
				//Upper triangular elements satisfy this condition. So, this can be used to zero out U entries in DpL
				const int gnbr_elem = isdistributed ? m->gglobalElemIndex(nbr_elem) : nbr_elem;
				ierr = MatSetValuesBlocked(DpL, 1, &element, 1, &gnbr_elem, zeros, INSERT_VALUES); CHKERRQ(ierr);

			}
		}


	}

	MatAssemblyBegin(DpL, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
	MatAssemblyEnd(DpL, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);

	MatAssemblyBegin(DpU, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
	MatAssemblyEnd(DpU, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);

	return ierr;
}


template <int nvars,typename scalar>
PetscErrorCode MatrixFreePreconditiner<nvars,scalar>::m_LUSGS(PC pc, Vec x, Vec y)
{

	//Details of the Linear system solved for PC:
	//PC is M = (D+L)D^{-1}(D+U), D+L+U = A; All are block matrices.
	//The preconditioning step is y = M^{-1}x
	//So we solve: (D+L)D^{-1}(D+U)y = x
	//Let D^{-1}(D+U)y = z or (D+U)y = Dz
	//So we have two sets of lin systems: (D+L)z = x and (D+U)y = Dz
	//Forward Sweep: Solve (D+L)z = x for z
	//Backward Sweep: Solve (D+U)y = Dz for y

	// Currently uses Richardson + PCILU for forward and backward sweeps. 
	// The overall non-lin convergence time is average for shell pc this way when compared to bjacobi+sor as pc for lin system in aodeSolver.cpp it but is manageable as this is needed temprarily. 
	// Change settings for Richardson and PCILU or use any other combos as needed IN THIS FUNCTION ITSELF.

	PetscErrorCode ierr = 0;
	Vec z;
	ierr = VecDuplicate(u, &z);CHKERRQ(ierr);


	//Forward sweep
	KSP forward;
	PC forward_pc;

	ierr = KSPCreate(PETSC_COMM_WORLD, &forward);CHKERRQ(ierr);
	ierr = KSPSetType(forward, KSPRICHARDSON);CHKERRQ(ierr);
	ierr = KSPGetPC(forward, &forward_pc);CHKERRQ(ierr);
	ierr = PCSetType(forward_pc, PCILU);CHKERRQ(ierr);
	PCType pctype;
	ierr = PCGetType(forward_pc, &pctype);CHKERRQ(ierr);
	ierr = KSPSetTolerances(forward, 0.5, PETSC_DEFAULT, PETSC_DEFAULT, 5); CHKERRQ(ierr);
	ierr = KSPSetOperators(forward, DpL, DpL);CHKERRQ(ierr);
	ierr = KSPSolve(forward, x, z);CHKERRQ(ierr); // z = (D+L)^{-1}x

	//Backward sweep
	KSP backward;
	PC backward_pc;

	Vec temp;
	ierr = VecDuplicate(u, &temp);CHKERRQ(ierr);
	ierr = MatMult(D, z, temp);CHKERRQ(ierr); // temp = Dz

	ierr = KSPCreate(PETSC_COMM_WORLD, &backward);CHKERRQ(ierr);
	ierr = KSPSetType(backward, KSPRICHARDSON);CHKERRQ(ierr);
	ierr = KSPGetPC(backward, &backward_pc);CHKERRQ(ierr);
	ierr = PCSetType(backward_pc, PCILU);CHKERRQ(ierr);
	ierr = KSPSetTolerances(backward, 0.5, PETSC_DEFAULT, PETSC_DEFAULT, 5); CHKERRQ(ierr);

	ierr = KSPSetOperators(backward, DpU, DpU);CHKERRQ(ierr);
	ierr = KSPSolve(backward, temp, y);CHKERRQ(ierr); // y = (D+U)^{-1}Dz

	//Destroy all the temporary vectors and KSPs
	ierr = VecDestroy(&z);CHKERRQ(ierr);
	ierr = VecDestroy(&temp);CHKERRQ(ierr);
	ierr = KSPDestroy(&forward);CHKERRQ(ierr);
	ierr = KSPDestroy(&backward);CHKERRQ(ierr);
	writePetscObj(y, "ym");
	//std::abort();
	return ierr;
}


template <int nvars,typename scalar>
PetscErrorCode MatrixFreePreconditiner<nvars,scalar>::setup_shell_pc_mf_lusgs(PC pc)
{
	PetscErrorCode ierr = 0;
	Mat A;
	ierr = PCGetOperators(pc, NULL, &A);CHKERRQ(ierr);
	ierr = MatDuplicate(A, MAT_DO_NOT_COPY_VALUES, &Dinv);CHKERRQ(ierr);
	ierr = MatDuplicate(A, MAT_DO_NOT_COPY_VALUES, &D);CHKERRQ(ierr);
	ierr = MatInvertBlockDiagonalMat(A,Dinv);CHKERRQ(ierr);

	ierr = MatAssemblyBegin(Dinv, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
	ierr = MatAssemblyEnd(Dinv, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

	//Calculate fluxes at state u.
	const UMesh<freal,NDIM> *const m = space->mesh();
	const std::vector<fint> globindices = m->getConnectivityGlobalIndices();

	ierr = VecCreateGhostBlock(PETSC_COMM_WORLD, nvars, (m->gninface()+m->gnbface())*nvars,
	                           m->gnaface()*nvars, m->gnConnFace(),
	                           globindices.data(), &fluxvec);CHKERRQ(ierr);

	ierr = space->assemble_fluxvec(u, fluxvec);CHKERRQ(ierr);
	ierr = VecGhostUpdateBegin(fluxvec, INSERT_VALUES, SCATTER_FORWARD);CHKERRQ(ierr);
	ierr = VecGhostUpdateEnd(fluxvec, INSERT_VALUES, SCATTER_FORWARD);CHKERRQ(ierr);
	//ierr = get_block_LUD(A);CHKERRQ(ierr);
	return ierr;
}

#if 1
template <int nvars,typename scalar>
PetscErrorCode MatrixFreePreconditiner<nvars,scalar>::t_mf_LUSGS(PC pc, Vec x, Vec y)
{
	//Testing: L*vec = flux(u+epsilon*vec) - flux(uvec) when the diff is taken only at left elements

	PetscErrorCode ierr = 0;

	Vec v;
	ierr = VecDuplicate(u, &v);CHKERRQ(ierr);
	ierr = VecSet(v, 0.0);CHKERRQ(ierr);
	//ierr = VecSetValue(v, 10, 0.01, INSERT_VALUES);CHKERRQ(ierr);
	ierr = VecSet(v, 0.01);CHKERRQ(ierr);
	ierr = VecAssemblyBegin(v);CHKERRQ(ierr);
	ierr = VecAssemblyEnd(v);CHKERRQ(ierr);
	
	// ierr = VecAssemblyBegin(v);CHKERRQ(ierr);
	// ierr = VecAssemblyEnd(v);CHKERRQ(ierr);
	//writePetscObj(v, "v");
	//ierr = VecShift(v, 0.1);CHKERRQ(ierr); 
	PetscRandom rctx;
    ierr = PetscRandomCreate(PETSC_COMM_WORLD, &rctx);CHKERRQ(ierr);
	unsigned long seed = 69;
	ierr = PetscRandomSetSeed(rctx, seed);CHKERRQ(ierr);
    ierr = PetscRandomSetFromOptions(rctx);CHKERRQ(ierr);
    ierr = VecSetRandom(v, rctx);CHKERRQ(ierr);
    ierr = PetscRandomDestroy(&rctx);CHKERRQ(ierr);
	//ierr = VecShift(v, 0.1);CHKERRQ(ierr); //ensures non-zero vec
	//ierr = VecScale(v, 0.01);CHKERRQ(ierr);
	//writePetscObj(v, "v");

	PetscScalar nrm2,nrm1;PetscInt size;
	ierr = VecGetSize(v, &size);CHKERRQ(ierr);
	ierr = VecNorm(v, NORM_2, &nrm2);CHKERRQ(ierr);
	ierr = VecNorm(u, NORM_1, &nrm1);CHKERRQ(ierr);
	PetscScalar epsilon = 1e-6;
	PetscScalar pertmag = epsilon*nrm1/(size*nrm2)+epsilon;
	//std::cout<<"Pertmag: "<<pertmag<<std::endl;

	PetscScalar tol = 1e-6;
	pertmag = tol/nrm2;
	//pertmag = 1.0;
	std::cout<<"Pertmag: "<<pertmag<<std::endl;
	Mat L;
	ierr = MatDuplicate(DpL, MAT_COPY_VALUES, &L);CHKERRQ(ierr);
	ierr = MatAXPY(L, -1.0, D,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr); //L = L-D
	//writePetscObj(L, "L");

	Vec aprod;
	ierr = VecDuplicate(u, &aprod);CHKERRQ(ierr);
	ierr = MatMult(L, v,aprod);CHKERRQ(ierr); //aprod = L*v

	MPI_Comm mycomm;
	ierr = PetscObjectGetComm((PetscObject)u, &mycomm); CHKERRQ(ierr);
	const int mpisize = get_mpi_size(mycomm);
	const bool isdistributed = (mpisize > 1);

	Vec z,upert,prod;
	ierr = VecDuplicate(u, &z);CHKERRQ(ierr);
	ierr = VecDuplicate(u, &upert);CHKERRQ(ierr);

	ierr = VecDuplicate(u, &prod);CHKERRQ(ierr); //Test
	ierr = VecWAXPY(upert, pertmag, v, u);CHKERRQ(ierr); //upert = u + pertmag*v

  
	const UMesh<freal,NDIM> *const m = space->mesh();
	Vec pertflux; //perturbed flux vector at a given face.
	ierr = VecDuplicate(fluxvec, &pertflux);CHKERRQ(ierr);
	// ierr = VecCreate(PETSC_COMM_WORLD, &pertflux);CHKERRQ(ierr);
	// ierr = VecSetSizes(pertflux,nvars, PETSC_DETERMINE);
	ierr = space->assemble_fluxvec(upert, pertflux);CHKERRQ(ierr);//Test
	ierr = VecGhostUpdateBegin(fluxvec, INSERT_VALUES, SCATTER_FORWARD);CHKERRQ(ierr);
	ierr = VecGhostUpdateEnd(fluxvec, INSERT_VALUES, SCATTER_FORWARD);CHKERRQ(ierr);

	for(fint i = 0; i < m->gnelem(); i++)
	{
		const fint element = isdistributed ? m->gglobalElemIndex(i) : i; //Global index number of element in case of parallel run
		int nface = m->gnfael(i); //Number of faces of the element

		PetscScalar sum[nvars];
		ierr = PetscArrayzero(sum,nvars);CHKERRQ(ierr);

		PetscInt idx[nvars];
		std::iota(idx, idx + nvars, element * nvars);
		
		for(int jface=0; jface<nface ; jface++)
		{
			int nbr_elem = m->gesuel(element,jface); //Neighbour element

			if (nbr_elem >=m->gnelem()) 
				continue;
			
			if(nbr_elem < element) // if mat = 4x4, and i = 3 here, lower triangle elements are all <3. That is the logic.
			{
				// We are at L part
				//TODO: Take diff pertflux and fluxvec here and put it inside prod. The check is Lv = prod or not.
				const fint faceID = m->gelemface(element,jface); 

				for(int k = 0; k<nvars; k++)
				{
					PetscScalar pertfluxval, fluxvecval;
					int id = faceID*nvars+k;
					ierr = VecGetValues(pertflux, 1, &id, &pertfluxval);CHKERRQ(ierr);
					ierr = VecGetValues(fluxvec, 1, &id, &fluxvecval);CHKERRQ(ierr);
					
					sum[k] += -(pertfluxval - fluxvecval)/pertmag;
				}
				
			}
		}


		ierr = VecSetValues(prod, nvars, idx, sum, INSERT_VALUES);CHKERRQ(ierr);
		ierr = VecAssemblyBegin(prod);CHKERRQ(ierr);
		ierr = VecAssemblyEnd(prod);CHKERRQ(ierr);
	}



	

	writePetscObj(aprod, "aprod");
	writePetscObj(prod, "calprod");
	// std::abort();
	return ierr;

}
#endif


template <int nvars,typename scalar>
PetscErrorCode MatrixFreePreconditiner<nvars,scalar>::mf_LUSGS(PC pc, Vec x, Vec y)
{
	//Todo:Just do the epsilon thing here and check man. ffs.
	//!Details from Debugging:------------
	//> Don't look at flux anymore. The flux vectors are correctly assembled. They were tested by calculating the residual at u using FVENS original method and using assemble_flux_vec method. Both gave same results for a non-zero u.
	
	//std::cout<<"In MF LUSGS"<<std::endl;
	using Eigen::Matrix; using Eigen::RowMajor;

	PetscErrorCode ierr = 0;
	//ierr = m_LUSGS(pc, x, y);CHKERRQ(ierr);
	//writePetscObj(x, "x");
	MPI_Comm mycomm;
	ierr = PetscObjectGetComm((PetscObject)u, &mycomm); CHKERRQ(ierr);
	const int mpisize = get_mpi_size(mycomm);
	const bool isdistributed = (mpisize > 1);

	const UMesh<freal,NDIM> *const m = space->mesh();

	Vec z,upert;
	ierr = VecDuplicate(u, &z);CHKERRQ(ierr);
	ierr = VecDuplicate(u, &upert);CHKERRQ(ierr);
	ierr = VecCopy(u, upert);CHKERRQ(ierr);
	ierr = VecSet(z,0.0);CHKERRQ(ierr);
	
	Vec pertflux;
	ierr = VecCreate(PETSC_COMM_WORLD, &pertflux);CHKERRQ(ierr);
	ierr = VecSetType(pertflux,VECSEQ);CHKERRQ(ierr);
	ierr = VecSetSizes(pertflux,nvars, nvars);CHKERRQ(ierr);
	
	//Forward Sweep
	PetscScalar tol = 1e-6;
	PetscScalar pertmag = tol;

	for(fint i = 0; i < m->gnelem(); i++)
	{
		const fint element = isdistributed ? m->gglobalElemIndex(i) : i; //Global index number of element in case of parallel run
		int nface = m->gnfael(i); //Number of faces of the element

		//PetscScalar sum[nvars];
		Eigen::VectorXd sum;
		sum.setZero(nvars);
		
		//ierr = PetscArrayzero(sum,nvars);CHKERRQ(ierr);

		PetscInt idx[nvars];
		std::iota(idx, idx + nvars, element * nvars);
		// std::cout<<"-------------------------------"<<std::endl;
		// std::cout<<i<<std::endl;
		//PetscBool islower = PETSC_FALSE;
		//ierr = VecNorm(z, NORM_2, &znrm);CHKERRQ(ierr);
		
		//pertmag = (znrm < tol)? tol : tol/znrm;
		for(int jface=0; jface<nface ; jface++)
		{
			int nbr_elem = m->gesuel(element,jface); //Neighbour element
			

			if (nbr_elem >=m->gnelem()) 
				continue;
			
			// This is the \sum_{j:j<i} (f(u+z)-f(z)) part
			if(nbr_elem < element) // if mat = 4x4, and i = 3 here, lower triangle elements are all <3. That is the logic.
			{
				// We are at L part
				//islower = PETSC_TRUE;
				const fint faceID = m->gelemface(element,jface); 
				// std::cout<<"-------------------------------"<<std::endl;
				// std::cout<<i<<std::endl;
				 
				// std::cout<<"L nbr_elem: "<<nbr_elem<<std::endl;
				// std::cout<<"face: "<<faceID<<std::endl;
				//Get Fluxes
				ierr = VecSet(pertflux, 0.0);CHKERRQ(ierr);
				ierr = space->assemble_fluxes_face(upert,pertflux,faceID);CHKERRQ(ierr);
				ierr = VecAssemblyBegin(pertflux);CHKERRQ(ierr);
				ierr = VecAssemblyEnd(pertflux);CHKERRQ(ierr);

				for(int k = 0; k<nvars; k++)
				{
					PetscScalar pertfluxval, fluxvecval;
					int id = faceID*nvars+k;
					ierr = VecGetValues(pertflux, 1, &k, &pertfluxval);CHKERRQ(ierr);
					if(std::isnan(pertfluxval))
					{
						std::cout<<i<<std::endl;
						
						std::abort();
					}
					ierr = VecGetValues(fluxvec, 1, &id, &fluxvecval);CHKERRQ(ierr);
					sum[k] += -(pertfluxval - fluxvecval)/pertmag; //!Should it be negative here and positive at U based on flux calc coz it seems to give better results like this? 
				}
				
			}
		}

		
		Eigen::VectorXd xval(nvars);
		ierr = VecGetValues(x, nvars, idx, xval.data());CHKERRQ(ierr);
		xval = xval - sum;

		//Perform D^{-1}(x_i - sum_{j:j<i} (f(u+z)-f(z))
		Matrix<freal,nvars,nvars,RowMajor> Dinv_elem;
		ierr = MatGetValues(Dinv,nvars,idx,nvars,idx,Dinv_elem.data());CHKERRQ(ierr);

		//PetscScalar zval[nvars];
		Eigen::VectorXd zval(nvars);
		zval = Dinv_elem*xval; //zval = D_i^{-1}(x_i - sum_{j:j<i} (f(u+z)-f(z))

		//std::cout<<zval<<std::endl;

		ierr = VecSetValues(z, nvars, idx, zval.data(), INSERT_VALUES);CHKERRQ(ierr);
		zval = zval*pertmag;
		ierr = VecSetValues(upert, nvars, idx, zval.data(), ADD_VALUES);CHKERRQ(ierr); //Update upert = u+z;

		

		// ierr = VecGetValues(upert, nvars, idx, zval.data());CHKERRQ(ierr);
		// //std::cout<<i<<std::endl;
		// std::cout<<zval<<std::endl;

	}

	ierr = VecAssemblyBegin(z);CHKERRQ(ierr);
	ierr = VecAssemblyEnd(z);CHKERRQ(ierr);
	ierr = VecAssemblyBegin(upert);CHKERRQ(ierr);
	ierr = VecAssemblyEnd(upert);CHKERRQ(ierr);
	
	

	//Backward Sweep
	ierr = VecCopy(u,upert);CHKERRQ(ierr);//Setting upert back to u
	for(int i = m->gnelem() - 1; i >= 0; i--)
	{
		const fint element = isdistributed ? m->gglobalElemIndex(i) : i; //Global index number of element in case of parallel run
		int nface = m->gnfael(i); //Number of faces of the element

		Eigen::VectorXd sum;
		sum.setZero(nvars);

		PetscInt idx[nvars];
		std::iota(idx, idx + nvars, element * nvars);
		// PetscScalar ynrm;
		// ierr = VecNorm(y, NORM_2, &ynrm);CHKERRQ(ierr);
		// pertmag = (ynrm < tol)? tol : tol/ynrm;
		for(int jface=0; jface<nface ; jface++)
		{
			int nbr_elem = m->gesuel(element,jface); //Neighbour element

			if (nbr_elem >=m->gnelem()) 
				continue;
			
			// This is the \sum_{j:j>i} (f(u+z)-f(z)) part
			if((nbr_elem > element)) // if mat = 4x4, and i = 3 here, upper triangle elements are all > 3. That is the logic.
			{
				// We are at U part
				const fint faceID = m->gelemface(element,jface); 
				
				//Get Fluxes
				ierr = VecSet(pertflux, 0.0);CHKERRQ(ierr);
				ierr = space->assemble_fluxes_face(upert,pertflux,faceID);CHKERRQ(ierr);
				ierr = VecAssemblyBegin(pertflux);CHKERRQ(ierr);
				ierr = VecAssemblyEnd(pertflux);CHKERRQ(ierr);

				for(int k = 0; k<nvars; k++)
				{
					PetscScalar pertfluxval, fluxvecval;
					int id = faceID*nvars+k;
					ierr = VecGetValues(pertflux, 1, &k, &pertfluxval);CHKERRQ(ierr);
					ierr = VecGetValues(fluxvec, 1, &id, &fluxvecval);CHKERRQ(ierr);

					sum[k] += (pertfluxval - fluxvecval)/pertmag;
				}

			}
		}

		Matrix<freal,nvars,nvars,RowMajor> Dinv_elem;
		ierr = MatGetValues(Dinv,nvars,idx,nvars,idx,Dinv_elem.data());CHKERRQ(ierr); //elemental Dinv matrix

		Eigen::VectorXd zval(nvars),yval(nvars);
		ierr = VecGetValues(z, nvars, idx, zval.data());CHKERRQ(ierr); //get z values at the given element

		yval = zval - Dinv_elem*sum; //y = z - D_i^{-1}[sum_{j:j>i} (f(u+z)-f(z))]

		ierr = VecSetValues(y, nvars, idx, yval.data(), INSERT_VALUES);CHKERRQ(ierr);
		yval = yval*pertmag;
		ierr = VecSetValues(upert, nvars, idx, yval.data(), ADD_VALUES);CHKERRQ(ierr); //Update upert = u+y;

	}
	ierr = VecAssemblyBegin(y);CHKERRQ(ierr);
	ierr = VecAssemblyEnd(y);CHKERRQ(ierr);
	ierr = VecAssemblyBegin(upert);CHKERRQ(ierr);
	ierr = VecAssemblyEnd(upert);CHKERRQ(ierr);


	//writePetscObj(y, "ymf");
	//std::abort();
	//std::cout<<"Done MF LUSGS\n";
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
PetscErrorCode MatrixFreePreconditiner<nvars,scalar>::setup_shell_pc(PC pc)
{
	PetscErrorCode ierr = 0;
	ierr = setup_shell_pc_mf_lusgs(pc);CHKERRQ(ierr);
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

	ierr = mfpc->mf_LUSGS(pc, x, y); //Select function manually for now
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
	ierr = PCShellGetContext(pc, &mfpc);CHKERRQ(ierr);

	delete mfpc;		
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
	const std::string namefin = name + ".m";
	//PetscCall(VecView(v, PETSC_VIEWER_STDOUT_WORLD));

	PetscCall(PetscPrintf(PETSC_COMM_WORLD, "writing vector ...\n"));
	PetscCall(PetscViewerASCIIOpen(PETSC_COMM_WORLD, namefin.c_str(), &viewer));
	PetscCall(PetscViewerPushFormat(viewer, PETSC_VIEWER_ASCII_MATLAB));
	PetscCall(VecView(v, viewer));
	PetscCall(PetscViewerDestroy(&viewer));
	return 0;

} 



} // namespace fvens
