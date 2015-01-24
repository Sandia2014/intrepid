// -*- C++ -*-
// matrixMultiplication.cc
// a huge comparison of doing naive and tiled matrix multiplication using many
//  different methods and technologies

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <string>
#include <algorithm>

// yucky, but for asking the system how many cores we have
#include <unistd.h>
#include <assert.h>

// header file for openmp
#include <omp.h>

// header files for kokkos
#include <Kokkos_Core.hpp>

/*
#include "Teuchos_Array.hpp"
#include "Intrepid_ArrayTools.hpp"
#include "Intrepid_FieldContainer.hpp"
#include "Intrepid_RealSpaceTools.hpp"
#include "Teuchos_oblackholestream.hpp"
#include "Teuchos_RCP.hpp"
#include "Teuchos_ScalarTraits.hpp"
#include "Teuchos_GlobalMPISession.hpp"
*/
#include <cuda_runtime.h>

using std::string;
using std::vector;
//using Intrepid::FieldContainer;

//typedef Intrepid::RealSpaceTools<double> rst;

#define BLOCK_SIZE 64;

//Pre-C++11 timing (thanks jeff)
double getElapsedTime(const timespec start, const timespec end) {
  timespec temp;
  if ((end.tv_nsec-start.tv_nsec)<0) {
    temp.tv_sec = end.tv_sec-start.tv_sec-1;
    temp.tv_nsec = 1000000000+end.tv_nsec-start.tv_nsec;
  } else {
    temp.tv_sec = end.tv_sec-start.tv_sec;
    temp.tv_nsec = end.tv_nsec-start.tv_nsec;
  }
  return double(temp.tv_sec) + double(temp.tv_nsec) / 1e9;
}


double random_double() {
    return (double) rand();
}


    
void contractFieldFieldTensorSerial(double * outputFields,
                                    double *   leftFields,
                                    double *  rightFields,
                                    const bool sumInto,
				    int numCells,
				    int numLeftFields,
				    int numRightFields,
				    int numPoints,
				    int dim1Tensor,
				    int dim2Tensor) {
  /*
  // get sizes
  int numCells        = leftFields.dimension(0);
  int numLeftFields   = leftFields.dimension(1);
  int numRightFields  = rightFields.dimension(1);
  int numPoints       = leftFields.dimension(2);
  int dim1Tensor      = leftFields.dimension(3);
  int dim2Tensor      = leftFields.dimension(4);
  */
    if (sumInto) {
	for (int cl = 0; cl < numCells; cl++) {
	    int clOff = numLeftFields*numPoints*dim1Tensor*dim2Tensor;
	    int crOff = numRightFields*numPoints*dim1Tensor*dim2Tensor;
	    int cOut = numLeftFields*numRightFields;
	    for (int lbf = 0; lbf < numLeftFields; lbf++) {
		int lOff = numPoints*dim1Tensor*dim2Tensor;
		int lOut = numRightFields;
		for (int rbf = 0; rbf < numRightFields; rbf++) {
		    int tmpVal = 0;
		    int rOff = numPoints*dim1Tensor*dim2Tensor;
		    for (int qp = 0; qp < numPoints; qp++) {
			int pOff = dim1Tensor*dim2Tensor;
			for (int iTens1 = 0; iTens1 < dim1Tensor; iTens1++) {
			    int tenOff = dim2Tensor;
			    for (int iTens2 = 0; iTens2 < dim2Tensor; iTens2++) {
				tmpVal +=
				leftFields[cl*clOff+lbf*lOff+qp*pOff+iTens1*tenOff+iTens2] *
				rightFields[cl*crOff+rbf*rOff+qp*pOff+iTens1*tenOff+iTens2];
			    } // D2-loop
			} // D1-loop
		    } // P-loop
		    outputFields[cl*cOut+lbf*lOut+rbf] += tmpVal;
		} // R-loop
	    } // L-loop
	} // C-loop
    }
    else {
	for (int cl = 0; cl < numCells; cl++) {
	    int clOff = numLeftFields*numPoints*dim1Tensor*dim2Tensor;
	    int crOff = numRightFields*numPoints*dim1Tensor*dim2Tensor;
	    int cOut = numLeftFields*numRightFields;
	    for (int lbf = 0; lbf < numLeftFields; lbf++) {
		int lOff = numPoints*dim1Tensor*dim2Tensor;
		int lOut = numRightFields;
		for (int rbf = 0; rbf < numRightFields; rbf++) {
		    int tmpVal = 0;
		    int rOff = numPoints*dim1Tensor*dim2Tensor;
		    for (int qp = 0; qp < numPoints; qp++) {
			int pOff = dim1Tensor*dim2Tensor;
			for (int iTens1 = 0; iTens1 < dim1Tensor; iTens1++) {
			    int tenOff = dim2Tensor;
			    for (int iTens2 = 0; iTens2 < dim2Tensor; iTens2++) {
				tmpVal +=
				leftFields[cl*clOff+lbf*lOff+qp*pOff+iTens1*tenOff+iTens2] *
				rightFields[cl*crOff+rbf*rOff+qp*pOff+iTens1*tenOff+iTens2];
			    } // D2-loop
			} // D1-loop
		    } // P-loop
		    outputFields[cl*cOut+lbf*lOut+rbf] = tmpVal;
		} // R-loop
	    } // L-loop
	} // C-loop
   }
} // end contractFieldFieldTensor
 

template<class DeviceType, class LeftViewType, class RightViewType, class
OutputViewType>
struct contractFieldFieldTensorFunctor {
    typedef DeviceType device_type;
    LeftViewType _leftFields;
    RightViewType _rightFields;
    OutputViewType _outputFields;
    int _numCells;
    int _numLeftFields;
    int _numRightFields;
    int _numPoints;
    int _dim1Tens;
    int _dim2Tens;

    contractFieldFieldTensorFunctor(LeftViewType leftFields, RightViewType
    rightFields, OutputViewType outputFields, int numCells, int numLeftFields, int
    numRightFields, int numPoints, int dim1Tens, int dim2Tens) :
    _leftFields(leftFields), _rightFields(rightFields),
    _outputFields(outputFields), _numCells(numCells), _numPoints(numPoints),
    _numLeftFields(numLeftFields), _numRightFields(numRightFields),
    _dim1Tens(dim1Tens), _dim2Tens(dim2Tens)
    {

    }

    KOKKOS_INLINE_FUNCTION
	void operator() (const unsigned int elementIndex) const {
	    int myID = elementIndex;

	    if(myID < (_numCells * _numLeftFields * _numRightFields)) {
		int myCell = myID / (_numLeftFields * _numRightFields);
		int matrixIndex = myID % (_numLeftFields * _numRightFields);

		
		int lbf = matrixIndex / _numRightFields;
		int rbf = matrixIndex % _numRightFields;
		/*
		int sub = _dim1Tens * _dim2Tens;
		int left1 = myCell* _numLeftFields* _numCells;
		int left2 = lbf* _numCells * sub;
		int left = myCell * _numLeftFields * _numCells * sub + lbf *
		_numCells * sub;
		int right = myCell * _numCells * sub * _numRightFields;
		int rsub = sub * _numRightFields;
		*/
		double temp = 0;
		for (int qp = 0; qp < _numPoints; qp++) {
		    for (int iTens1 = 0; iTens1 < _dim1Tens; iTens1++) {
			for (int iTens2 = 0; iTens2 < _dim2Tens; iTens2++) {
			    temp += _leftFields(myCell, lbf, qp,
				iTens1,
				iTens2) *
				_rightFields(myCell, qp,
				iTens1,
				iTens2,
				rbf);
			}
		    }
		}
		_outputFields(myCell, lbf, rbf)= temp;
	    }
	}

};



template <class DeviceType, class input_view_t, class output_view_t, class
input_host_t, class output_host_t>
void contractFieldFieldTensorKokkos(output_host_t& outHost,
    const input_host_t & leftHost,
    const input_host_t & rightHost,
    output_view_t & outDevice,
    input_view_t & leftDevice,
    input_view_t & rightDevice,
    int numCells,
    int numLeftFields,
    int numRightFields,
    int numPoints,
    int dim1Tens,
    int dim2Tens,
    double* time = NULL) {
    
    /*
    int numCells = leftHost.dimension(0);
    int numLeftFields = leftHost.dimension(1);
    int numRightFields = rightHost.dimension(4);
    int numPoints = leftHost.dimension(2);
    */

    Kokkos::deep_copy(leftDevice, leftHost);
    Kokkos::deep_copy(rightDevice, rightHost);
    Kokkos::deep_copy(outDevice, outHost);

    timespec tic;
    if (time != NULL) {
	clock_gettime(CLOCK_MONOTONIC, &tic);
    }

    contractFieldFieldTensorFunctor<DeviceType, input_view_t, input_view_t,
    output_view_t> kokkosFunctor(leftDevice, rightDevice, outDevice,
    numCells, numLeftFields, numRightFields, numPoints, dim1Tens, dim2Tens);

    Kokkos::parallel_for(numCells*numLeftFields*numRightFields, kokkosFunctor);

    Kokkos::fence();

    timespec toc;
    if (time != NULL) {
	clock_gettime(CLOCK_MONOTONIC, &toc);
	*time += getElapsedTime(tic, toc);
    }

    Kokkos::deep_copy(outHost, outDevice);



}

int main(int argc, char* argv[]) {
  int c=10000, p=10, l=10, r=10, t1=10, t2=10;
  int cLOff = l*p*t1*t2;
  int cROff = r*p*t1*t2;
  int basisOff = p*t1*t2;
  int pLOff = t1*t2;
  int pROff = t1*t2*r;
  int tROff = t2*r;
  int t2ROff = r;
  int tOff = t2;

printf("Nowisht\n");
    double * in_c_l_p_t1_t2 = new double[c*l*p*t1*t2];
    double * in_c_r_p_t1_t2 = new double[c*r*p*t1*t2];
    double * out1_c_l_r = new double[c*l*r];
    double * out2_c_l_r = new double[c*l*r];
  //double zero = Intrepid::INTREPID_TOL*100000.0;
  double zero = 0;

    printf("here\n");

    for (int cl = 0; cl < c; ++cl) {
	int cOff = p * t1 * t2 * r;
	for(int rbf = 0; rbf < r; ++rbf) {
	    int rOff = p*t1*t2;
	    for (int qp = 0; qp < p; ++qp) {
		int pOff = t1*t2;
		for (int iTens1 = 0; iTens1 < t1; ++iTens1) {
		    int t1Off = t2;
		    for (int iTens2 = 0; iTens2 < t2; ++iTens2) {
			in_c_r_p_t1_t2[cl*cOff+rbf*rOff+qp*pOff+iTens1*t1Off+iTens2]
			= random_double();
		    }
		}
	    }
	}
    }

    for (int cl = 0; cl < c; ++cl) {
	int cOff = p*t1*t2*l;
	for(int lbf = 0; lbf < l; ++lbf) {
	    int lOff = p*t1*t2;
	    for (int qp = 0; qp < p; ++qp) {
		int pOff = t1*t2;
		for (int iTens1 = 0; iTens1 < t1; ++iTens1) {
		    int t1Off = t2;
		    for (int iTens2 = 0; iTens2 < t2; ++iTens2) {
			in_c_l_p_t1_t2[cl*cOff+lbf*lOff+qp*pOff+iTens1*t1Off+iTens2]
			= random_double();
		    }
		}
	    }
	}
    }



    for (int cl = 0; cl < c; cl++) {
	for (int lbf = 0; lbf < l; lbf++) {
	    for (int rbf = 0; rbf < r; rbf++) {
		out1_c_l_r[cl*r*l + lbf*r + rbf] = 0;
		out2_c_l_r[cl*r*l + lbf*r + rbf] =0;

	    }
	}
    }


    /*
  // fill with random numbers
  for (int i=0; i<c*l*p*t1*t2; i++) {
    in_c_l_p_t1_t2[i] = random_double();
  }
  for (int i=0; i<c*r*p*t1*t2; i++) {
    in_c_r_p_t1_t2[i] = random_double();
  }
  */
  std::cout << "Created vectors" << std::endl;

    timespec tic;
    clock_gettime(CLOCK_MONOTONIC, &tic);

    contractFieldFieldTensorSerial(out1_c_l_r, in_c_l_p_t1_t2, in_c_r_p_t1_t2, 
    false, c, l, r, p, t1, t2);

    timespec toc;
    clock_gettime(CLOCK_MONOTONIC, &toc);
    const double elapsedTime_serial = getElapsedTime(tic, toc);

    std::cout << "serial elapsed time: " << elapsedTime_serial << " sec" <<
    std::endl;
  

    /* HERE IS WHERE I HAVE CHECK EVERYTHING */

  // ===============================================================
  // ********************** < Kokkos setup> ************************
  // vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv

  // Doing all of this here might throw off the timing -- we're not counting the
  // cost of the copy into Kokkos or the deep copy from Kokkos host to Kokkos
  // device.



  //Cuda arrays
/*
  double * cudaRight = new double[c * r * p * t1 * t2];
  double * cudaLeft = new double[c * l * p * t1 * t2];
  double * cudaOut = new double[c * l * r];


  for (int cl = 0; cl < c; ++cl) {
    for (int qp = 0; qp < p; ++qp) {
      for (int iTens1 = 0; iTens1 < t1; ++iTens1) {
        for (int iTens2 = 0; iTens2 < t2; ++iTens2) {
          for(int rbf = 0; rbf < r; ++rbf) {
            cudaRight[cl * p * t1 * t2 * r +
                      qp * t1 * t2 * r +
                      iTens1 * t2 * r +
                      iTens2 * r +
                      rbf] = in_c_r_p_t1_t2[cl][rbf][qp][iTens1][iTens2];
          }
          for(int lbf = 0; lbf < l; ++lbf) {
            cudaLeft[cl * l * p * t1 * t2 +
                     lbf * p * t1 * t2 +
                     qp * t1 * t2 +
                     iTens1 * t2 +
                     iTens2] = in_c_l_p_t1_t2[cl][lbf][qp][iTens1][iTens2];
          }
        }
      }
    }
  }



  std::cout << "trying serial" << std::endl;
    


    timespec tic;
    clock_gettime(CLOCK_MONOTONIC, &tic);

    timespec toc;
    clock_gettime(CLOCK_MONOTONIC, &toc);

  
  //Warmup
  contractFieldFieldScalarSerial(out2_c_l_r, in_c_l_p_t1_t2, in_c_r_p_t1_t2);

  timespec tic;
  clock_gettime(CLOCK_MONOTONIC, &tic);

  //repeat the calculation 5 times so we can average out some randomness
  for(int i = 0; i < 5; ++i){
    contractFieldFieldScalarSerial(out2_c_l_r, in_c_l_p_t1_t2, in_c_r_p_t1_t2);
  }

  timespec toc;
  clock_gettime(CLOCK_MONOTONIC, &toc);
  const double elapsedTime_serial = getElapsedTime(tic, toc);

  std::cout << "serial took " << elapsedTime_serial << " second" << std::endl;
  */


/*
  std::cout << "trying cuda" << std::endl;
  //Now try the cuda version, start with warmup
  cudaDocontractFieldFieldScalar(cudaOut,cudaLeft,cudaRight, c, l, r, p, t1, t2, &tic, &toc);
  double elapsedTime_cuda = 0;
  for(int i = 0; i < 5; ++i){
    cudaDocontractFieldFieldScalar(cudaOut,cudaLeft,cudaRight, c, l, r, p, t1, t2, &tic, &toc);
    elapsedTime_cuda += getElapsedTime(tic,toc);
  }

  for (int cl = 0; cl < c; ++cl) {
    for(int lbf = 0; lbf < l; ++lbf) {
      for(int rbf = 0; rbf < r; ++rbf) {
        out1_c_l_r(cl,lbf,rbf) = cudaOut[cl * l * r + lbf * r + rbf];
      }
    }
  }

  rst::subtract(&out1_c_l_r[0], &out2_c_l_r[0], out2_c_l_r.size());
  if (rst::vectorNorm(&out1_c_l_r[0], out1_c_l_r.size(), Intrepid::NORM_ONE) > zero) {
    std::cout << "\n\nINCORRECT contractFieldFieldTensor (1): check cuda; "
    << " diff-1norm = " << rst::vectorNorm(&out1_c_l_r[0], out1_c_l_r.size(), Intrepid::NORM_ONE) << "\n\n";
  }

  std::cout << "cuda speedup of " << elapsedTime_serial/elapsedTime_cuda << std::endl;
*/

    Kokkos::initialize();

    typedef Kokkos::View<double *****, Kokkos::LayoutRight, Kokkos::Cuda>
    cuda_input_view_t;
    typedef Kokkos::View<double ***, Kokkos::LayoutRight, Kokkos::Cuda>
    cuda_output_view_t;
    typedef typename cuda_input_view_t::HostMirror cuda_input_host_t;
    typedef typename cuda_output_view_t::HostMirror cuda_output_host_t;

    typedef Kokkos::View<double *****, Kokkos::LayoutRight, Kokkos::OpenMP>
    omp_input_view_t;
    typedef Kokkos::View<double *****, Kokkos::LayoutRight, Kokkos::OpenMP>
    omp_output_view_t;


    cuda_input_view_t cuda_kokkosLeft("left_input", c, l, p, t1, t2);
    cuda_input_view_t cuda_kokkosRight("right_input", c, p, t1, t2, r);
    cuda_output_view_t cuda_kokkosOut("output", c, l, r);

    cuda_input_host_t cuda_hostLeft("left_input", c, l, p, t1, t2);
    cuda_input_host_t cuda_hostRight("left_input", c, p, t1, t2, r);
    cuda_output_host_t cuda_hostOut("left_input", c, l, r);

    printf("filling views\n");

    for (int cl = 0; cl < c; ++cl) {
	for (int qp = 0; qp < p; ++qp) {
	    for (int iTens1 = 0; iTens1 < t1; ++iTens1) {
		for (int iTens2 = 0; iTens2 < t2; ++iTens2) {
		    for(int rbf = 0; rbf < r; ++rbf) {
			cuda_hostRight(cl,
			    qp, 
			    iTens1,
			    iTens2,
			    rbf) =
			    in_c_r_p_t1_t2[cl*cROff+rbf+qp*pROff+iTens1*tROff+iTens2*t2ROff];
		    }
		    for(int lbf = 0; lbf < l; ++lbf) {
			cuda_hostLeft(cl, 
			    lbf,
			    qp,
			    iTens1,
			    iTens2) =
			    in_c_l_p_t1_t2[cl*cLOff+lbf*basisOff+qp*pLOff+iTens1*tOff+iTens2];
		    }
		}
	    }
	}
    }
    printf("trying Kokkos Cuda\n");
    
    /*
    // THIS NEEDS HELP!
    contractFieldFieldTensorKokkos<Kokkos::Cuda, cuda_input_view_t,
    cuda_output_view_t, cuda_input_host_t, cuda_output_host_t>(cuda_hostOut,
    cuda_hostLeft, cuda_hostRight, cuda_kokkosOut, cuda_kokkosLeft,
    cuda_kokkosRight, c, l, r, p, t1, t2);
    clock_gettime(CLOCK_MONOTONIC, &tic);
    */

    printf("Done with warmup\n");
    
    double elapsedTime_kokkos_cuda_nocopy = 0;
    
    clock_gettime(CLOCK_MONOTONIC, &tic);

    contractFieldFieldTensorKokkos<Kokkos::Cuda, cuda_input_view_t,
	cuda_output_view_t, cuda_input_host_t, cuda_output_host_t>(cuda_hostOut,
	cuda_hostLeft, cuda_hostRight, cuda_kokkosOut, cuda_kokkosLeft,
        cuda_kokkosRight, c, l, r, p, t1, t2, &elapsedTime_kokkos_cuda_nocopy);
    
    clock_gettime(CLOCK_MONOTONIC, &toc);

    elapsedTime_kokkos_cuda_nocopy = getElapsedTime(tic, toc);



    for (int i = 0; i < c; c++) {
	for (int j = 0; j < l; j++) {
	    for (int k = 0; k < r; k++) {
		double diff = cuda_hostOut(i, j, k) - out1_c_l_r[i*l*r +
		j*r + k];
		if (diff < 0) {
		    diff = -diff;
		}
		double frac = cuda_hostOut(i, j, k)/100;
		if (frac < 0) {
		    frac = -frac;
		}
		if (diff > frac) {
		    std::cout << "we have a problem" << std::endl;
		    std::cout << i << " " << j << " " << k << std::endl;
		    std::cout << "serial num " << out1_c_l_r[i*l*r +j*r +k] <<
		    std::endl;
		    std::cout << "para num " << cuda_hostOut(i, j, k) <<
		    std::endl;
		    Kokkos::finalize();
		    return 0;
		}
	    }
	}
    }

    /*

    for (int i = 0; i < 5; i++) {
	// Do 5 times
	contractFieldFieldTensorKokkos<Kokkos::Cuda, cuda_input_view_t,
	    cuda_output_view_t, cuda_input_host_t, cuda_output_host_t>(cuda_hostOut,
	    cuda_hostLeft, cuda_hostRight, cuda_kokkosOut, cuda_kokkosLeft,
	    cuda_kokkosRight, c, l, r, p, t1, t2, &elapsedTime_kokkos_cuda_nocopy);
    }
    clock_gettime(CLOCK_MONOTONIC, &toc);
 
    
    for (int i = 0; i < out2_c_l_r.size(); i++) {
	// The indexing needs help on this problem!!!!
	double diff = cuda_kokkosOut(i%c, fig out l, fig out r) - out2_c_l_r(i);
	if (diff < 0) {
	    diff = diff *(-1);
	}
	if (diff > cuda_kokkosOut(i)*.0000001) {
	    printf("Error in Compute\n");
	}
    }
    */

    std::cout << "kokkos runtime of " << elapsedTime_kokkos_cuda_nocopy << std::endl;
    std::cout << "speed up of " <<
    elapsedTime_serial/elapsedTime_kokkos_cuda_nocopy << std::endl;
    Kokkos::finalize();

  return 0;
}
