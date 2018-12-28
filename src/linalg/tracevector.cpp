/** \file
 * \brief Implementation of trace vectors
 *
 * This file is part of FVENS.
 *   FVENS is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   FVENS is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with FVENS.  If not, see <http://www.gnu.org/licenses/>.
 */

#include <set>
#include "utilities/mpiutils.hpp"
#include "tracevector.hpp"

namespace fvens {

template <typename scalar, int nvars>
L2TraceVector<scalar,nvars>::L2TraceVector(const UMesh2dh<scalar>& mesh) : m{mesh}
{
	update_comm_pattern();
}

template <typename scalar, int nvars>
void L2TraceVector<scalar,nvars>::update_comm_pattern()
{
	left.resize(m.gnaface()*nvars);
	right.resize(m.gnaface()*nvars);

	sharedfacemap.resize(m.gnConnFace(),1);

	std::set<int> nbds;
	for(a_int icface = 0; icface < m.gnConnFace(); icface++)
	{
		nbds.insert(m.gconnface(icface,2));
	}
	nbdranks.resize(nbds.size());
	std::copy(nbds.begin(), nbds.end(), nbdranks.begin());

#ifdef DEBUG
	const int mpirank = get_mpi_rank(MPI_COMM_WORLD);
	std::cout << "Rank " << mpirank << ": nbd = " << nbdranks.size() << std::endl;
	std::cout << "Rank " << mpirank << ":    nbds = ";
	for(size_t i= 0; i < nbdranks.size(); i++) {
		std::cout << nbdranks[i] << " ";
	}
	std::cout << std::endl;
#endif

	for(a_int icface = 0; icface < m.gnConnFace(); icface++)
	{
		bool matched = false;
		for(int i = 0; i < static_cast<int>(nbdranks.size()); i++)
			if(m.gconnface(icface,2) == nbdranks[i]) {
				sharedfacemap(icface,0) = i;
				matched = true;
				break;
			}
		assert(matched);
	}

	sharedfaces.resize(nbdranks.size());
	for(a_int icface = 0; icface < m.gnConnFace(); icface++)
	{
		sharedfaces[sharedfacemap(icface,0)].push_back(icface);
	}

#ifdef DEBUG
	// check face counts
	{
		a_int totface = 0;
		for(size_t irank = 0; irank < nbdranks.size(); irank++)
			totface += sharedfaces[irank].size();
		assert(totface == m.gnConnFace());
	}

	// check sharedfaces against connface
	for(size_t irank = 0; irank < nbdranks.size(); irank++)
		for(size_t isface = 0; isface < sharedfaces[irank].size(); isface++)
			assert(nbdranks[irank] == m.gconnface(sharedfaces[irank][isface],2));

	// check sharedfacemap
	for(a_int icface = 0; icface < m.gnConnFace(); icface++)
		assert(nbdranks[sharedfacemap(icface,0)] == m.gconnface(icface,2));
#endif

	sharedfacesother.resize(nbdranks.size());
	for(size_t irank = 0; irank < nbdranks.size(); irank++) {
		sharedfacesother[irank].assign(sharedfaces[irank].size(), -1);
	}

	std::vector<std::vector<a_int>> nbdGFaceIndex(nbdranks.size());
	for(size_t irank = 0; irank < nbdranks.size(); irank++)
		nbdGFaceIndex[irank].resize(sharedfaces[irank].size());

	std::vector<MPI_Request> rrequests(nbdranks.size());

	// Send and receive element indices corresponding to shared faces
	{
		std::vector<MPI_Request> srequests(nbdranks.size());
		std::vector<std::vector<a_int>> myGFaceIndex(nbdranks.size());

		for(size_t irank = 0; irank < nbdranks.size(); irank++)
		{
			myGFaceIndex[irank].resize(sharedfaces[irank].size());
			for(size_t iface = 0; iface < sharedfaces[irank].size(); iface++)
				myGFaceIndex[irank][iface] = m.gconnface(sharedfaces[irank][iface],4);

			const int dest = nbdranks[irank];
			const int tag = irank;
			MPI_Isend(&myGFaceIndex[irank][0], myGFaceIndex[irank].size(), FVENS_MPI_INT, dest, tag,
			          MPI_COMM_WORLD, &srequests[irank]);
		}

		for(size_t irank = 0; irank < nbdranks.size(); irank++)
		{
			const int source = nbdranks[irank];
			//const int tag = irank;
			MPI_Irecv(&nbdGFaceIndex[irank][0], nbdGFaceIndex[irank].size(), FVENS_MPI_INT,
			          source, MPI_ANY_TAG,
			          MPI_COMM_WORLD, &rrequests[irank]);
		}

		// Wait for sends to finish
		for(size_t irank = 0; irank < nbdranks.size(); irank++)
			MPI_Wait(&srequests[irank], MPI_STATUS_IGNORE);
	}

	// Wait for receives to finish
	for(size_t irank = 0; irank < nbdranks.size(); irank++)
		MPI_Wait(&rrequests[irank], MPI_STATUS_IGNORE);

	std::cout << "Setup: received" << std::endl;

	// set the mapping between connectivity faces and receive buffers
	for(a_int icface = 0; icface < m.gnConnFace(); icface++)
	{
		const int rankindex = sharedfacemap(icface,0);
		assert(nbdranks[rankindex] == m.gconnface(icface,2));

		bool matched = false;

		for(a_int isface = 0; isface < static_cast<a_int>(sharedfaces[rankindex].size()); isface++)
		{
			if(nbdGFaceIndex[rankindex][isface] == m.gconnface(icface,4))
			{
				sharedfacesother[rankindex][isface] = icface;
				matched = true;
				break;
			}
		}

		assert(matched);
	}

	std::cout << "Built communication pattern." << std::endl;

	for(size_t irank = 0; irank < nbdranks.size(); irank++)
		for(size_t isface = 0; isface < sharedfacesother[irank].size(); isface++)
			assert(sharedfacesother[irank][isface] != -1);

	std::cout << "All shared faces matched." << std::endl;
}

template <typename scalar, int nvars>
scalar *L2TraceVector<scalar,nvars>::getLocalArrayLeft()
{
	return &left[0];
}

template <typename scalar, int nvars>
scalar *L2TraceVector<scalar,nvars>::getLocalArrayRight()
{
	return &right[0];
}

template <typename scalar, int nvars>
const scalar *L2TraceVector<scalar,nvars>::getLocalArrayLeft() const
{
	return &left[0];
}

template <typename scalar, int nvars>
const scalar *L2TraceVector<scalar,nvars>::getLocalArrayRight() const
{
	return &right[0];
}

/** A more general implementation would use MPI_BYTE and manually serialize the array of scalars.
 * But the easier way would be to define templated MPI communication functions and use those below.
 */
template <typename scalar, int nvars>
void L2TraceVector<scalar,nvars>::updateSharedFacesBegin()
{
	std::vector<MPI_Request> sreq(nbdranks.size());
	std::vector<std::vector<scalar>> buffers(nbdranks.size());

	const a_int fstart = m.gConnBFaceStart();
	for(size_t irank = 0; irank < nbdranks.size(); irank++)
	{
		buffers[irank].resize(sharedfaces[irank].size()*nvars);
		for(size_t iface = 0; iface < sharedfaces[irank].size(); iface++)
		{
			for(int ivar = 0; ivar < nvars; ivar++)
				buffers[irank][iface*nvars+ivar]
					= left [(sharedfaces[irank][iface]+fstart)*nvars + ivar];
		}

		MPI_Isend(&buffers[irank][0], sharedfaces.size()*nvars, FVENS_MPI_REAL, nbdranks[irank], irank,
		          MPI_COMM_WORLD, &sreq[irank]);
	}

	std::vector<MPI_Status> status(nbdranks.size());
	MPI_Waitall(nbdranks.size(), &sreq[0], &status[0]);
}

template <typename scalar, int nvars>
void L2TraceVector<scalar,nvars>::updateSharedFacesEnd()
{
	std::vector<MPI_Request> rreq(nbdranks.size());
	std::vector<std::vector<scalar>> buffers(nbdranks.size());

	for(size_t irank = 0; irank < nbdranks.size(); irank++)
	{
		buffers[irank].resize(sharedfaces[irank].size()*nvars);
		MPI_Irecv(&buffers[irank][0], sharedfaces.size()*nvars, FVENS_MPI_REAL, nbdranks[irank], irank,
		          MPI_COMM_WORLD, &rreq[irank]);
	}

	const a_int fstart = m.gConnBFaceStart();
	for(size_t irank = 0; irank < nbdranks.size(); irank++)
	{
		MPI_Wait(&rreq[irank], MPI_STATUS_IGNORE);

		for(size_t isface = 0; isface < sharedfaces[irank].size(); isface++)
		{
			for(int ivar = 0; ivar < nvars; ivar++)
				right[(fstart+sharedfacesother[irank][isface])*nvars+ivar]
					= buffers[irank][isface*nvars+ivar];
		}
	}

}

template class L2TraceVector<a_real,1>;
template class L2TraceVector<a_real,NVARS>;

}
