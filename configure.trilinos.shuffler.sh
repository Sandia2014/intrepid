#!/bin/sh

#----------------------------------------------------------------------
# Clean up cmake and make generated files:

rm -rf CMake* Trilinos* packages Dart* Testing cmake_install.cmake MakeFile*

TRILINOS_ROOT="/home/jeff/sandia14"

#----------------------------------------------------------------------
# For debug build:

# cmake	\
#	-D CMAKE_BUILD_TYPE:STRING=DEBUG	\
#	-D Kokkos_ENABLE_BOUNDS_CHECK:BOOL=ON	\
#	\

# For optimized build:

# cmake	\
#	-D CMAKE_BUILD_TYPE:STRING=RELEASE	\

#----------------------------------------------------------------------
# Options to CUDA_NVCC_FLAGS must be semi-colon delimited,
# this is different than the standard CMAKE_CXX_FLAGS syntax.

CUDA_ARCH="30"
CUDA_FLAGS="-DKOKKOS_HAVE_CUDA_ARCH=${CUDA_ARCH}0;-gencode;arch=compute_${CUDA_ARCH},code=sm_${CUDA_ARCH}"
CUDA_FLAGS="${CUDA_FLAGS};--Werror;cross-execution-space-call"
CUDA_FLAGS="${CUDA_FLAGS};-Xcudafe;--diag_suppress=code_is_unreachable"
CUDA_FLAGS="${CUDA_FLAGS};-Xcompiler;-Wall,-ansi"
CUDA_FLAGS="${CUDA_FLAGS};-O3"

cmake	\
	-D CMAKE_BUILD_TYPE:STRING=RELEASE	\
	\
	-D TPL_ENABLE_Pthread:BOOL=ON	\
	-D TPL_ENABLE_HWLOC:BOOL=ON	\
	\
  	-D TPL_ENABLE_CUDA:BOOL=ON	\
	-D TPL_ENABLE_CUSPARSE:BOOL=ON	\
	-D CUDA_NVCC_FLAGS:STRING="${CUDA_FLAGS}"	\
	\
	-D Trilinos_ENABLE_ALL_PACKAGES:BOOL=OFF	\
	-D Trilinos_ENABLE_EXAMPLES:BOOL=ON	\
	-D Trilinos_ENABLE_TESTS:BOOL=ON	\
	\
	-D Trilinos_ENABLE_KokkosCore:BOOL=ON	\
	-D Trilinos_ENABLE_KokkosContainers:BOOL=ON	\
	-D Trilinos_ENABLE_KokkosLinAlg:BOOL=ON	\
	-D Trilinos_ENABLE_KokkosExample:BOOL=ON	\
	-D Kokkos_ENABLE_Pthread:BOOL=ON	\
	\
	-D Trilinos_ENABLE_Intrepid:BOOL=ON	\
	-D TPL_BLAS_LIBRARIES:STRING=/usr/lib/libblas.so.3	\
	-D TPL_LAPACK_LIBRARIES:STRING=/usr/lib/liblapack.so.3	\
	\
	-D CMAKE_INSTALL_PREFIX=${TRILINOS_ROOT}/trilinos-install	\
	\
	${TRILINOS_ROOT}/trilinos-11.10.2-Source

