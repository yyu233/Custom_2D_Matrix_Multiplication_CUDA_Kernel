#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

extern void cu_mmul(float* a, float* b, float* c, int M, int N ,int P);

namespace py = pybind11;

py::array_t<float> cu_mmul_wrapper(py::array_t<float> m1, py::array_t<float> m2) {

	auto buf1 = m1.request();
	auto buf2 = m2.request();

	if (m1.ndim() != 2 || m2.ndim() != 2) {
		throw std::runtime_error("Number of dimensions must be 2");
	}

	if (m1.shape()[1] != m2.shape()[0]) {
		throw std::runtime_error("The column dimension of first matrix must match the row dimension of the second matrix")
	}

	// M x O
	int M = m1.shape()[0];
	int N = m1.shape()[1];
	int P = m2.shape()[1];

	auto res = py::array(py::buffer_info(
		nullptr, 
		sizeof(float), 
		py::format_descriptor<float>::value, 
		2, 
		{M, P}, 
		{sizeof(float)*P, sizeof(float)}
		));

	auto buf3 = res.request();

	float* A = (float*)buf1.ptr;
	float* B = (float*)buf2.ptr;
	float* C = (float*)buf3.ptr;

	cu_mmul(A, B, C, M, N, P);

	return res;
}

PYBIND11_MODULE(cu_matrix_multiply, m) {
	m.doc() = "CUDA Matrix Mutiplication";

	m.def("mmul", &cu_mmul_wrapper, "2D Matrix multiplication");
}