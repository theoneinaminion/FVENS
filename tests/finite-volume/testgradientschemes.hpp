/** \file
 * \brief Tests for gradient schemes
 * \author Aditya Kashi
 */

#ifndef FVENS_TESTS_GRADIENTS_H
#define FVENS_TESTS_GRADIENTS_H

#include <string>
#include "spatial/aspatial.hpp"

namespace fvens_tests {

using fvens::freal;
using fvens::fint;
using Eigen::Matrix;
using Eigen::RowMajor;

/// A 'spatial discretization' that does nothing but carry out tests on gradient schemes etc
class TestSpatial : public fvens::Spatial<freal,1>
{
public:
	TestSpatial(const fvens::UMesh<freal,NDIM> *const mesh);

	fvens::StatusCode compute_residual(const Vec u, Vec residual,
	                                   const bool gettimesteps, Vec dtm) const
	{ return 0; }

	void compute_local_jacobian_interior(const fint iface,
	                                     const freal *const ul, const freal *const ur,
	                                     Matrix<freal,1,1,RowMajor>& L,
	                                     Matrix<freal,1,1,RowMajor>& U) const
	{ }

	void compute_local_jacobian_boundary(const fint iface,
	                                     const freal *const ul,
	                                     Matrix<freal,1,1,RowMajor>& L) const
	{ }

	virtual void getGradients(const Vec u,
	                          fvens::GradBlock_t<freal,NDIM,1> *const grads) const
	{ }

	/// Test if weighted least-squares reconstruction is '1-exact'
	int test_oneExact(const std::string reconst_type) const;

	//................. For Matrix-free LU-SGS preconditioning..................//
	/**
	 * @brief Assembles flux vector for the whole domain at each face taking into consideration
	 * the reconstructions, boundaries etc.
	 * 
	 * @param[in] uvec 
	 * @param[out] fluxvec 
	 * @return StatusCode 
	 */
	fvens::StatusCode assemble_fluxvec(const Vec uvec, Vec fluxvec) const{return 0;}	
	/**
	 * @brief Assemble flux at a given faceID
	 * 
	 * @param[in] uvec state vector 
	 * @param[out] fluxvec flux vector at a given face 
	 * @param[in] faceID Face ID 
	 * @return StatusCode 
	 */
	fvens::StatusCode assemble_fluxes_face(const Vec uvec, Vec fluxvec, const int faceID) const{return 0;}

protected:
	using fvens::Spatial<freal,1>::m;
	using fvens::Spatial<freal,1>::rch;
	using fvens::Spatial<freal,1>::gr;
};

}
#endif
