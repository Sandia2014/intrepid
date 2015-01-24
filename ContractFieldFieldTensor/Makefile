all: ContractFieldFieldScalar ContractFieldFieldTensor ContractDataDataScalar ScalarMultiplyDataField

ScalarMultiplyDataField: ScalarMultiplyDataField.cu
	/opt/cuda/bin/nvcc -ccbin=g++ -m64 -O3 -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35 -I/usr/local/cuda-6.5/include -I/clinic/2014/sandia14/trilinos-install-shadowfax//include -o ScalarMultiplyDataField ScalarMultiplyDataField.cu -Xcompiler -fopenmp -L/clinic/2014/sandia14/trilinos-install/lib -lkokkoscore -lhwloc -lgomp -lcudart -lcublas -lteuchoscore

ContractFieldFieldScalar: ContractFieldFieldScalar.cu
	/opt/cuda/bin/nvcc -ccbin=g++ -m64 -O3 -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35 -I/usr/local/cuda-6.5/include -I/clinic/2014/sandia14/trilinos-install-shadowfax//include -o ContractFieldFieldScalar ContractFieldFieldScalar.cu -Xcompiler -fopenmp -L/clinic/2014/sandia14/trilinos-install-shadowfax/lib -lkokkoscore -lhwloc -lgomp -lcudart -lcublas

ContractFieldFieldTensor: ContractFieldFieldTensor.cu
	/opt/cuda/bin/nvcc -ccbin=g++ -m64 -O3 -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35 -I/usr/local/cuda-6.5/include -I/clinic/2014/sandia14/trilinos-install-shadowfax//include -o ContractFieldFieldTensor ContractFieldFieldTensor.cu -Xcompiler -fopenmp -L/clinic/2014/sandia14/trilinos-install-shadowfax/lib -lkokkoscore -lhwloc -lgomp -lcudart -lcublas

ContractDataDataScalar: ContractDataDataScalar.cu
	/opt/cuda/bin/nvcc -ccbin=g++ -m64 -O3 -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35 -I/usr/local/cuda-6.5/include -I/clinic/2014/sandia14/trilinos-install//include -o ContractDataDataScalar ContractDataDataScalar.cu -Xcompiler -fopenmp -L/clinic/2014/sandia14/trilinos-install/lib -lkokkoscore -lhwloc -lgomp -lcudart -lcublas -lteuchoscore

clean:
	rm -f ContractFieldFieldScalar
	rm -f ContractDataDataScalar
	rm -f ContractFieldFieldTensor
