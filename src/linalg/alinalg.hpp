/** @file alinalg.hpp
 * @brief Setup and handling of some linear algebra objects 
 * @author Aditya Kashi
 */

#ifndef ALINALG_H
#define ALINALG_H

#include "aconstants.hpp"
#include "mesh/mesh.hpp"
#include "spatial/aspatial.hpp"

#include <petscksp.h>

#define PETSCOPTION_STR_LEN 30
#define USER_PC TRUE

#ifdef USE_BLASTED
#include <blasted_petsc.h>
#endif

namespace fvens {

/// Sets up global vectors with storage with ghost locations for connectivity boundaries
/** \param[in] nvars Number of physical variables per grid location
 */
StatusCode createGhostedSystemVector(const UMesh<freal,NDIM> *const m, const int nvars, Vec *const v);

/// Creates a Vec with one unknown per mesh cell
/** Currently, a 'sequential' Vec is created.
 */
StatusCode createSystemVector(const UMesh<freal,NDIM> *const m, const int nvars, Vec *const v);

/// Sets up storage preallocation for sparse matrix formats
/** \param[in] m Mesh context
 * \param[in|out] A The matrix to pre-allocate for
 *
 * We assume there's only 1 neighboring cell that's not in this subdomain
 * \todo TODO: Once a partitioned mesh is used, set the preallocation properly.
 *
 * The type of sparse matrix is read from the PETSc options database, but
 * only MPI matrices are supported.
 */
template <int nvars>
StatusCode setupSystemMatrix(const UMesh<freal,NDIM> *const m, Mat *const A);

/// Computes the amount of memory to be reserved for the Jacobian matrix
template <int nvars>
StatusCode setJacobianPreallocation(const UMesh<freal,NDIM> *const m, Mat A);

/// Matrix-free Jacobian of the flux
/** An object of this type is associated with a specific spatial discretization context.
 * The normalized step length epsilon for the finite-difference Jacobian is set to a default value,
 * but it also queried from the PETSc options database.
 */
template <int nvars>
class MatrixFreeSpatialJacobian
{
public:
	/// Query the Petsc options database for a custom option for setting the step length epsilon
	/** The finite difference step length epsilon is given a default value.
	 * \param[in] spatial_discretization The spatial discretization of which this objact
	 *   will act as Jacobian
	 */
	MatrixFreeSpatialJacobian(const Spatial<freal,nvars> *const spatial_discretization);

	/// Set the state u at which the Jacobian is computed, the corresponding residual r(u) and 
	/// the diagonal vector of the mass matrix for each cell
	/** Note that the residual vector supplied is assumed to be the negative of what is needed,
	 * exactly what Spatial::compute_residual gives.
	 */
	int set_state(const Vec u_state, const Vec r_state, const Vec mdts);

	/// Compute a Jacobian-vector product
	StatusCode apply(const Vec x, Vec y) const;

protected:
	/// Spatial discretization context
	const Spatial<freal,nvars> *const spatial;

	/// step length for finite difference Jacobian
	freal eps;

	/// The state at which to compute the Jacobian
	Vec u;

	/// The residual of the state \ref uvec at which to compute the Jacobian
	Vec res;

	/// Time steps for each cell
	Vec mdt;
};

/// Setup a matrix-free Mat for the Jacobian
/** Sets up the matrix-free Jacobian with the spatial discretization as well as sets up the PETSc
 * shell Mat with it.
 * \param[in] spatial The spatial discretization context to associate with the matrix-free Jacobian
 * \param[out] A The Mat to setup
 */
template <int nvars>
StatusCode create_matrixfree_jacobian(const Spatial<freal,nvars> *const spatial, Mat *const A);

/// Returns true iff the argument is a matrix-free PETSc Mat
bool isMatrixFree(Mat);

#ifdef USE_BLASTED

/// Sets BLASTed preconditioners
/** Only one subdomain per rank is supported in case of domain decomposition global preconditioners.
 * \param ksp The top-level KSP
 * \param u A solution vector used to assemble the Jacobian matrix once, needed for initialization
 *   of some PETSc preconditioners - the actual values don't matter.
 * \param startprob A spatial discretization context to compute the Jacobian with
 * \param[in,out] bctx The BLASTed contexts
 */
template <int nvars>
StatusCode setup_blasted(KSP ksp, Vec u, const Spatial<freal,nvars> *const startprob,
                         Blasted_data_list& bctx);

#endif


//#ifdef USER_PC

template <int nvars,typename scalar>
class MatrixFreePreconditiner
{
	/**
	 * @brief This is supposed to be an abstract class that would serve as the base class for all user-defined petsc preconditioners. 
	 * 
	 */
	public:
	/**
	 * @brief Construct a new Matrix Free Spatial Jacobian object
	 * 
	 * @param spatial_discretization the spatial discretization of which this object will act as preconditioner.
	 */
	MatrixFreePreconditiner(const Spatial<freal,nvars> *const spatial_discretization);

	/// Set the state u and the corresponding residual r(u) 
	void set_state(const Vec u_state, const Vec r_state);

	/**
	 * @brief Shell preconditioner apply function using matrix-version of LU-SGS
	 * 
	 * @param pc preconditioning context
	 * @param x  unpreconditioned vector
	 * @param y  preconditioned vector
	 * @return PetscErrorCode 
	 */
	PetscErrorCode m_LUSGS(PC pc, Vec x, Vec y);

	/**
	 * @brief Shell preconditioner apply function using matrix-free version of LU-SGS
	 * 
	 * @param pc preconditioning context
	 * @param x  unpreconditioned vector
	 * @param y  preconditioned vector
	 * @return PetscErrorCode 
	 */
	PetscErrorCode mf_LUSGS(PC pc, Vec x, Vec y);

	/**
	 * @brief Get the L, U, D parts of the Jacobian where A = L+D+U
	 * @param Mat the jacobina matrix
	 */
	PetscErrorCode get_LUD(Mat &A);

	PetscErrorCode testapply(PC pc, Vec x, Vec y) const;

	/**
	 * @brief Set the up shell pc object (wrapper)
	 * 
	 * @param pc 
	 * @return PetscErrorCode 
	 */
	PetscErrorCode setup_shell_pc(PC pc);

	/**
	 * @brief Set the up Data Structures required for Mattrix LUSGS preconditioner
	 * 
	 * @param pc 
	 * @return PetscErrorCode 
	 */
	PetscErrorCode setup_shell_pc_mlusgs(PC pc);

	protected:
	/// Spatial discretization context
	const Spatial<freal,nvars> *space;

	/// step length for finite difference Jacobian
	freal eps;

	//KSP context where the preconditioning happens
	//const KSP *const ksp;

	/// The state at which to compute the Jacobian
	Vec u;

	/// The residual of the state \ref uvec at which to compute the Jacobian
	Vec res;

	//Contituent matrices of the Jacobian A = L+D+U
	Mat DpL; //D+L
	Mat D;
	Mat DpU; //D+U
	Mat Dinv;

};


/**
 * @brief Shell preconditioner setup function
 * 
 * @param pc preconditioning context
 * @return PetscErrorCode 
 */
template <int nvars,typename scalar>
PetscErrorCode pcsetup(PC pc);

/**
 * @brief Shell preconditioner apply function
 * 
 * @param pc preconditioning context
 * @param x  unpreconditioned vector
 * @param y  preconditioned vector
 * @return PetscErrorCode 
 */
template <int nvars,typename scalar>
PetscErrorCode pcapply(PC pc, Vec x, Vec y);

/**
 * @brief Shell preconditioner destroy function
 * 
 * @param pc preconditioning context
 * @return PetscErrorCode 
 */
template <int nvars,typename scalar>
PetscErrorCode pcdestroy(PC pc);

//#endif

template <int nvars,typename scalar>
PetscErrorCode create_shell_precond(const Spatial<freal,nvars> *const spatial, PC *pc);

PetscErrorCode writePetscObj(Vec &v, std::string name); 

PetscErrorCode writePetscObj(Mat &A, std::string name);


}
#endif
