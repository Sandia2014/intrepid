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
#include "Teuchos_Array.hpp"
#include "Intrepid_ArrayTools.hpp"
#include "Intrepid_FieldContainer.hpp"
#include "Intrepid_RealSpaceTools.hpp"
#include "Teuchos_oblackholestream.hpp"
#include "Teuchos_RCP.hpp"
#include "Teuchos_ScalarTraits.hpp"
#include "Teuchos_GlobalMPISession.hpp"

#include <cuda_runtime.h>

using std::string;
using std::vector;
using Intrepid::FieldContainer;

typedef Intrepid::RealSpaceTools<double> rst;

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

template<class ArrayOutFields, class ArrayInData, class ArrayInFields>
void scalarMultiplyDataField(ArrayOutFields &     outputFields,
                                         const ArrayInData &  inputData,
                                         ArrayInFields &      inputFields,
                                         const bool           reciprocal) {

#ifdef HAVE_INTREPID_DEBUG
  TEUCHOS_TEST_FOR_EXCEPTION( (inputData.rank() != 2), std::invalid_argument,
                      ">>> ERROR (ArrayTools::scalarMultiplyDataField): Input data container must have rank 2.");
  if (outputFields.rank() <= inputFields.rank()) {
    TEUCHOS_TEST_FOR_EXCEPTION( ( (inputFields.rank() < 3) || (inputFields.rank() > 5) ), std::invalid_argument,
                        ">>> ERROR (ArrayTools::scalarMultiplyDataField): Input fields container must have rank 3, 4, or 5.");
    TEUCHOS_TEST_FOR_EXCEPTION( (outputFields.rank() != inputFields.rank()), std::invalid_argument,
                        ">>> ERROR (ArrayTools::scalarMultiplyDataField): Input and output fields containers must have the same rank.");
    TEUCHOS_TEST_FOR_EXCEPTION( (inputFields.dimension(0) != inputData.dimension(0) ), std::invalid_argument,
                        ">>> ERROR (ArrayTools::scalarMultiplyDataField): Zeroth dimensions (number of integration domains) of the fields and data input containers must agree!");
    TEUCHOS_TEST_FOR_EXCEPTION( ( (inputFields.dimension(2) != inputData.dimension(1)) && (inputData.dimension(1) != 1) ), std::invalid_argument,
                        ">>> ERROR (ArrayTools::scalarMultiplyDataField): Second dimension of the fields input container and first dimension of data input container (number of integration points) must agree or first data dimension must be 1!");
    for (int i=0; i<inputFields.rank(); i++) {
      std::string errmsg  = ">>> ERROR (ArrayTools::scalarMultiplyDataField): Dimension ";
      errmsg += (char)(48+i);
      errmsg += " of the input and output fields containers must agree!";
      TEUCHOS_TEST_FOR_EXCEPTION( (inputFields.dimension(i) != outputFields.dimension(i)), std::invalid_argument, errmsg );
    }
  }
  else {
    TEUCHOS_TEST_FOR_EXCEPTION( ( (inputFields.rank() < 2) || (inputFields.rank() > 4) ), std::invalid_argument,
                        ">>> ERROR (ArrayTools::scalarMultiplyDataField): Input fields container must have rank 2, 3, or 4.");
    TEUCHOS_TEST_FOR_EXCEPTION( (outputFields.rank() != inputFields.rank()+1), std::invalid_argument,
                        ">>> ERROR (ArrayTools::scalarMultiplyDataField): The rank of the input fields container must be one less than the rank of the output fields container.");
    TEUCHOS_TEST_FOR_EXCEPTION( ( (inputFields.dimension(1) != inputData.dimension(1)) && (inputData.dimension(1) != 1) ), std::invalid_argument,
                        ">>> ERROR (ArrayTools::scalarMultiplyDataField): First dimensions of fields input container and data input container (number of integration points) must agree or first data dimension must be 1!");
    TEUCHOS_TEST_FOR_EXCEPTION( ( inputData.dimension(0) != outputFields.dimension(0) ), std::invalid_argument,
                        ">>> ERROR (ArrayTools::scalarMultiplyDataField): Zeroth dimensions of fields output container and data input containers (number of integration domains) must agree!");
    for (int i=0; i<inputFields.rank(); i++) {
      std::string errmsg  = ">>> ERROR (ArrayTools::scalarMultiplyDataField): Dimensions ";
      errmsg += (char)(48+i);
      errmsg += " and ";
      errmsg += (char)(48+i+1);
      errmsg += " of the input and output fields containers must agree!";
      TEUCHOS_TEST_FOR_EXCEPTION( (inputFields.dimension(i) != outputFields.dimension(i+1)), std::invalid_argument, errmsg );
    }
  }
#endif

  // get sizes
  int invalRank      = inputFields.rank();
  int outvalRank     = outputFields.rank();
  int numCells       = outputFields.dimension(0);
  int numFields      = outputFields.dimension(1);
  int numPoints      = outputFields.dimension(2);
  int numDataPoints  = inputData.dimension(1);
  int dim1Tens       = 0;
  int dim2Tens       = 0;
  if (outvalRank > 3) {
    dim1Tens = outputFields.dimension(3);
    if (outvalRank > 4) {
      dim2Tens = outputFields.dimension(4);
    }
  }

  if (outvalRank == invalRank) {

    if (numDataPoints != 1) { // nonconstant data

      switch(invalRank) {
        case 3: {
          if (reciprocal) {
            for(int cl = 0; cl < numCells; cl++) {
              for(int bf = 0; bf < numFields; bf++) {
                for(int pt = 0; pt < numPoints; pt++) {
                  outputFields(cl, bf, pt) = inputFields(cl, bf, pt)/inputData(cl, pt);
                } // P-loop
              } // F-loop
            } // C-loop
          }
          else {
            for(int cl = 0; cl < numCells; cl++) {
              for(int bf = 0; bf < numFields; bf++) {
                for(int pt = 0; pt < numPoints; pt++) {
                  outputFields(cl, bf, pt) = inputFields(cl, bf, pt)*inputData(cl, pt);
                } // P-loop
              } // F-loop
            } // C-loop
          }
        }// case 3
        break;

        case 4: {
          if (reciprocal) {
            for(int cl = 0; cl < numCells; cl++) {
              for(int bf = 0; bf < numFields; bf++) {
                for(int pt = 0; pt < numPoints; pt++) {
                  for( int iVec = 0; iVec < dim1Tens; iVec++) {
                    outputFields(cl, bf, pt, iVec) = inputFields(cl, bf, pt, iVec)/inputData(cl, pt);
                  } // D1-loop
                } // P-loop
              } // F-loop
            } // C-loop
          }
          else {
            for(int cl = 0; cl < numCells; cl++) {
              for(int bf = 0; bf < numFields; bf++) {
                for(int pt = 0; pt < numPoints; pt++) {
                  for( int iVec = 0; iVec < dim1Tens; iVec++) {
                    outputFields(cl, bf, pt, iVec) = inputFields(cl, bf, pt, iVec)*inputData(cl, pt);
                  } // D1-loop
                } // P-loop
              } // F-loop
            } // C-loop
          }
        }// case 4
        break;

        case 5: {
          if (reciprocal) {
            for(int cl = 0; cl < numCells; cl++) {
              for(int bf = 0; bf < numFields; bf++) {
                for(int pt = 0; pt < numPoints; pt++) {
                  for( int iTens1 = 0; iTens1 < dim1Tens; iTens1++) {
                    for( int iTens2 = 0; iTens2 < dim2Tens; iTens2++) {
                      outputFields(cl, bf, pt, iTens1, iTens2) = inputFields(cl, bf, pt, iTens1, iTens2)/inputData(cl, pt);
                    } // D2-loop
                  } // D1-loop
                } // F-loop
              } // P-loop
            } // C-loop
          }
          else {
            for(int cl = 0; cl < numCells; cl++) {
              for(int bf = 0; bf < numFields; bf++) {
                for(int pt = 0; pt < numPoints; pt++) {
                  for( int iTens1 = 0; iTens1 < dim1Tens; iTens1++) {
                    for( int iTens2 = 0; iTens2 < dim2Tens; iTens2++) {
                      outputFields(cl, bf, pt, iTens1, iTens2) = inputFields(cl, bf, pt, iTens1, iTens2)*inputData(cl, pt);
                    } // D2-loop
                  } // D1-loop
                } // F-loop
              } // P-loop
            } // C-loop
          }
        }// case 5
        break;

        default:
	    ;
	}// invalRank

    }
    else { //constant data

      switch(invalRank) {
        case 3: {
          if (reciprocal) {
            for(int cl = 0; cl < numCells; cl++) {
              for(int bf = 0; bf < numFields; bf++) {
                for(int pt = 0; pt < numPoints; pt++) {
                  outputFields(cl, bf, pt) = inputFields(cl, bf, pt)/inputData(cl, 0);
                } // P-loop
              } // F-loop
            } // C-loop
          }
          else {
            for(int cl = 0; cl < numCells; cl++) {
              for(int bf = 0; bf < numFields; bf++) {
                for(int pt = 0; pt < numPoints; pt++) {
                  outputFields(cl, bf, pt) = inputFields(cl, bf, pt)*inputData(cl, 0);
                } // P-loop
              } // F-loop
            } // C-loop
          }
        }// case 3
        break;

        case 4: {
          if (reciprocal) {
            for(int cl = 0; cl < numCells; cl++) {
              for(int bf = 0; bf < numFields; bf++) {
                for(int pt = 0; pt < numPoints; pt++) {
                  for( int iVec = 0; iVec < dim1Tens; iVec++) {
                    outputFields(cl, bf, pt, iVec) = inputFields(cl, bf, pt, iVec)/inputData(cl, 0);
                  } // D1-loop
                } // P-loop
              } // F-loop
            } // C-loop
          }
          else {
            for(int cl = 0; cl < numCells; cl++) {
              for(int bf = 0; bf < numFields; bf++) {
                for(int pt = 0; pt < numPoints; pt++) {
                  for( int iVec = 0; iVec < dim1Tens; iVec++) {
                    outputFields(cl, bf, pt, iVec) = inputFields(cl, bf, pt, iVec)*inputData(cl, 0);
                  } // D1-loop
                } // P-loop
              } // F-loop
            } // C-loop
          }
        }// case 4
        break;

        case 5: {
          if (reciprocal) {
            for(int cl = 0; cl < numCells; cl++) {
              for(int bf = 0; bf < numFields; bf++) {
                for(int pt = 0; pt < numPoints; pt++) {
                  for( int iTens1 = 0; iTens1 < dim1Tens; iTens1++) {
                    for( int iTens2 = 0; iTens2 < dim2Tens; iTens2++) {
                      outputFields(cl, bf, pt, iTens1, iTens2) = inputFields(cl, bf, pt, iTens1, iTens2)/inputData(cl, 0);
                    } // D2-loop
                  } // D1-loop
                } // F-loop
              } // P-loop
            } // C-loop
          }
          else {
            for(int cl = 0; cl < numCells; cl++) {
              for(int bf = 0; bf < numFields; bf++) {
                for(int pt = 0; pt < numPoints; pt++) {
                  for( int iTens1 = 0; iTens1 < dim1Tens; iTens1++) {
                    for( int iTens2 = 0; iTens2 < dim2Tens; iTens2++) {
                      outputFields(cl, bf, pt, iTens1, iTens2) = inputFields(cl, bf, pt, iTens1, iTens2)*inputData(cl, 0);
                    } // D2-loop
                  } // D1-loop
                } // F-loop
              } // P-loop
            } // C-loop
          }
        }// case 5
        break;

        default:
	    ; 
      } // invalRank
    } // numDataPoints

  }
  else {

    if (numDataPoints != 1) { // nonconstant data

      switch(invalRank) {
        case 2: {
          if (reciprocal) {
            for(int cl = 0; cl < numCells; cl++) {
              for(int bf = 0; bf < numFields; bf++) {
                for(int pt = 0; pt < numPoints; pt++) {
                  outputFields(cl, bf, pt) = inputFields(bf, pt)/inputData(cl, pt);
                } // P-loop
              } // F-loop
            } // C-loop
          }
          else {
            for(int cl = 0; cl < numCells; cl++) {
              for(int bf = 0; bf < numFields; bf++) {
                for(int pt = 0; pt < numPoints; pt++) {
                  outputFields(cl, bf, pt) = inputFields(bf, pt)*inputData(cl, pt);
                } // P-loop
              } // F-loop
            } // C-loop
          }
        }// case 2
        break;

        case 3: {
          if (reciprocal) {
            for(int cl = 0; cl < numCells; cl++) {
              for(int bf = 0; bf < numFields; bf++) {
                for(int pt = 0; pt < numPoints; pt++) {
                  for( int iVec = 0; iVec < dim1Tens; iVec++) {
                    outputFields(cl, bf, pt, iVec) = inputFields(bf, pt, iVec)/inputData(cl, pt);
                  } // D1-loop
                } // P-loop
              } // F-loop
            } // C-loop
          }
          else {
            for(int cl = 0; cl < numCells; cl++) {
              for(int bf = 0; bf < numFields; bf++) {
                for(int pt = 0; pt < numPoints; pt++) {
                  for( int iVec = 0; iVec < dim1Tens; iVec++) {
                    outputFields(cl, bf, pt, iVec) = inputFields(bf, pt, iVec)*inputData(cl, pt);
                  } // D1-loop
                } // P-loop
              } // F-loop
            } // C-loop
          }
        }// case 3
        break;

        case 4: {
          if (reciprocal) {
            for(int cl = 0; cl < numCells; cl++) {
              for(int bf = 0; bf < numFields; bf++) {
                for(int pt = 0; pt < numPoints; pt++) {
                  for( int iTens1 = 0; iTens1 < dim1Tens; iTens1++) {
                    for( int iTens2 = 0; iTens2 < dim2Tens; iTens2++) {
                      outputFields(cl, bf, pt, iTens1, iTens2) = inputFields(bf, pt, iTens1, iTens2)/inputData(cl, pt);
                    } // D2-loop
                  } // D1-loop
                } // F-loop
              } // P-loop
            } // C-loop
          }
          else {
            for(int cl = 0; cl < numCells; cl++) {
              for(int bf = 0; bf < numFields; bf++) {
                for(int pt = 0; pt < numPoints; pt++) {
                  for( int iTens1 = 0; iTens1 < dim1Tens; iTens1++) {
                    for( int iTens2 = 0; iTens2 < dim2Tens; iTens2++) {
                      outputFields(cl, bf, pt, iTens1, iTens2) = inputFields(bf, pt, iTens1, iTens2)*inputData(cl, pt);
                    } // D2-loop
                  } // D1-loop
                } // F-loop
              } // P-loop
            } // C-loop
          }
        }// case 4
        break;

        default:
	    ;
	}// invalRank

    }
    else { //constant data

      switch(invalRank) {
        case 2: {
          if (reciprocal) {
            for(int cl = 0; cl < numCells; cl++) {
              for(int bf = 0; bf < numFields; bf++) {
                for(int pt = 0; pt < numPoints; pt++) {
                  outputFields(cl, bf, pt) = inputFields(bf, pt)/inputData(cl, 0);
                } // P-loop
              } // F-loop
            } // C-loop
          }
          else {
            for(int cl = 0; cl < numCells; cl++) {
              for(int bf = 0; bf < numFields; bf++) {
                for(int pt = 0; pt < numPoints; pt++) {
                  outputFields(cl, bf, pt) = inputFields(bf, pt)*inputData(cl, 0);
                } // P-loop
              } // F-loop
            } // C-loop
          }
        }// case 2
        break;

        case 3: {
          if (reciprocal) {
            for(int cl = 0; cl < numCells; cl++) {
              for(int bf = 0; bf < numFields; bf++) {
                for(int pt = 0; pt < numPoints; pt++) {
                  for( int iVec = 0; iVec < dim1Tens; iVec++) {
                    outputFields(cl, bf, pt, iVec) = inputFields(bf, pt, iVec)/inputData(cl, 0);
                  } // D1-loop
                } // P-loop
              } // F-loop
            } // C-loop
          }
          else {
            for(int cl = 0; cl < numCells; cl++) {
              for(int bf = 0; bf < numFields; bf++) {
                for(int pt = 0; pt < numPoints; pt++) {
                  for( int iVec = 0; iVec < dim1Tens; iVec++) {
                    outputFields(cl, bf, pt, iVec) = inputFields(bf, pt, iVec)*inputData(cl, 0);
                  } // D1-loop
                } // P-loop
              } // F-loop
            } // C-loop
          }
        }// case 3
        break;

        case 4: {
          if (reciprocal) {
            for(int cl = 0; cl < numCells; cl++) {
              for(int bf = 0; bf < numFields; bf++) {
                for(int pt = 0; pt < numPoints; pt++) {
                  for( int iTens1 = 0; iTens1 < dim1Tens; iTens1++) {
                    for( int iTens2 = 0; iTens2 < dim2Tens; iTens2++) {
                      outputFields(cl, bf, pt, iTens1, iTens2) = inputFields(bf, pt, iTens1, iTens2)/inputData(cl, 0);
                    } // D2-loop
                  } // D1-loop
                } // F-loop
              } // P-loop
            } // C-loop
          }
          else {
            for(int cl = 0; cl < numCells; cl++) {
              for(int bf = 0; bf < numFields; bf++) {
                for(int pt = 0; pt < numPoints; pt++) {
                  for( int iTens1 = 0; iTens1 < dim1Tens; iTens1++) {
                    for( int iTens2 = 0; iTens2 < dim2Tens; iTens2++) {
                      outputFields(cl, bf, pt, iTens1, iTens2) = inputFields(bf, pt, iTens1, iTens2)*inputData(cl, 0);
                    } // D2-loop
                  } // D1-loop
                } // F-loop
              } // P-loop
            } // C-loop
          }
        }// case 4
        break;

        default:
	    ;
	} // invalRank
    } // numDataPoints

  } // end if (outvalRank = invalRank)

}


int main(int argc, char* argv[]) {
    int c = 10000, p = 1000; //l = 10, r = 10, t1 = 10, t2 = 10;
    int b = 100;

    FieldContainer<double> in_Fields_3(c, b, p);
    FieldContainer<double> in_Data_2(c, p);

    FieldContainer<double> out_Fields3_Serial(c, b, p);
    FieldContainer<double> out_Fields3(c, b, p);

    for (int i = 0; i < in_Fields_3.size(); i++) {
	in_Fields_3[i] = Teuchos::ScalarTraits<double>::random();
    }
    for (int i = 0; i < in_Data_2.size(); i++) {
	in_Data_2[i] = Teuchos::ScalarTraits<double>::random();
    }
    std::cout << "Created the vectors" << std::endl;

    std::cout << "Trying serial" << std::endl;

    timespec tic;
    clock_gettime(CLOCK_MONOTONIC, &tic);

    scalarMultiplyDataField<FieldContainer<double>, FieldContainer<double>,
    FieldContainer<double> >(out_Fields3_Serial, in_Data_2, in_Fields_3, false);

    timespec toc;
    clock_gettime(CLOCK_MONOTONIC, &toc);
    const double elapsedTime_serial = getElapsedTime(tic, toc);

    std::cout << "serial took " << elapsedTime_serial << " seconds" <<
    std::endl;
    

    Kokkos::initialize();

    typedef Kokkos::View<double ***, Kokkos::LayoutRight, Kokkos::Cuda>
    cuda_input_fields_3;
    typedef Kokkos::View<double **, Kokkos::LayoutRight, Kokkos::Cuda>
    cuda_input_data_2;

    typedef Kokkos::View<double ***, Kokkos::LayoutRight, Kokkos::Cuda>
    cuda_output_fields_3;

    typedef typename cuda_input_fields_3::HostMirror cuda_input_fields_3_host;
    typedef typename cuda_input_data_2::HostMirror cuda_input_data_2_host;
    typedef typename cuda_output_fields_3::HostMirror cuda_output_fields_3_host;

    /*
    typedef Kokkos::View<double ***, Kokkos::LayoutRight, Kokkos::OpenMP>
    omp_input_view_t;
    typedef Kokkos::View<double ***, Kokkos::LayoutRight, Kokkos::OpenMP>
    omp_output_view_t;
    */

    cuda_input_fields_3 cuda_kokkosInputFields("input_Fields", c, b, p);
    cuda_input_data_2 cuda_kokkosInputData("input_Data", c, p);
    cuda_output_fields_3 cuda_kokkosOut("output", c, b, p);

    cuda_input_fields_3_host cuda_hostFields("left_input", c, b, p);
    cuda_input_data_2_host cuda_hostData("left_input", c, p);
    cuda_output_fields_3_host cuda_hostOut("left_input", c, b, p);

    printf("filling views\n");
    
    for (int cl = 0; cl < c; cl++) {
	for (int pt = 0; pt < p; pt++) {
	    for (int bf = 0; bf < b; bf++) {
		cuda_hostFields(cl, bf, pt) = in_Fields_3(cl, bf, pt);
	    }
	    cuda_hostData(cl, pt) = in_Data_2(cl, pt);
	}
    }
		
    //Now I need to call the function that will create the functor and run!


    Kokkos::finalize();

    return 0;

}








