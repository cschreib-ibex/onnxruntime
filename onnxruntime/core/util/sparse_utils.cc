// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/status.h"
#include "core/framework/tensor.h"
#include "core/framework/data_types_internal.h"
#include "core/framework/data_transfer_manager.h"
#include "core/framework/sparse_tensor.h"
#include "core/framework/sparse_crcsformat_rep.h"

#include "math_cpuonly.h"

#include <Eigen/SparseCore>

namespace onnxruntime {
namespace sparse_utils {
template <class T>
using SparseMatrixRow = Eigen::SparseMatrix<T, Eigen::RowMajor>;

template<typename In>
struct TypeMap {
  using Out = In;
};

template<>
struct TypeMap<MLFloat16> {
  using Out = Eigen::half;
};

template <typename T>
struct ToSparseConvert {
  Status operator()(const DataTransferManager& data_manager, const Tensor& src_cpu,
                    const AllocatorPtr& allocator, SparseTensor& dst) const {
    const auto* input_data = src_cpu.Data<T>();
    const auto& dense_shape = src_cpu.Shape();
    // We do not support a stack of matrices here
    assert(dense_shape.NumDimensions() == 2);
    auto M = dense_shape.GetDims()[0];
    auto N = dense_shape.GetDims()[1];

    ConstEigenMatrixMapRowMajor<TypeMap<T>::Out> dense_map(reinterpret_cast<const TypeMap<T>::Out*>(input_data), M, N);
    // Quick way to convert.
    SparseMatrixRow<TypeMap<T>::Out> sparse_matrix = dense_map.sparseView();
    sparse_matrix.makeCompressed();
    static_assert(sizeof(T) == sizeof(typename SparseMatrixRow<T>::Scalar), "Expecting data type parity");
    //static_assert(sizeof(int64_t) == sizeof(*sparse_matrix.innerIndexPtr()), "Expecting index type parity");

    const auto nnz = sparse_matrix.nonZeros();

    TensorShape values_shape{nnz};
    TensorShape inner_shape{nnz};
    TensorShape outer_shape{M + 1};
    Tensor values(src_cpu.DataType(), values_shape, sparse_matrix.valuePtr(), allocator->Info());
    Tensor inner_indices(DataTypeImpl::GetType<int64_t>(), inner_shape, sparse_matrix.innerIndexPtr(), allocator->Info());
    Tensor outer_indices(DataTypeImpl::GetType<int64_t>(), outer_shape, sparse_matrix.outerIndexPtr(), allocator->Info());

    SparseTensor sparse_tensor(src_cpu.DataType(), dense_shape, nnz, allocator);
    SparseCrcsFormatRep* rep = nullptr;
    auto builder = sparse_tensor.RepBuilder<SparseCrcsBuilder>();
    builder.GetOrCreate(SparseCrcsFormatRep::kRowMajor, inner_shape, outer_shape, rep);
    ORT_RETURN_IF_ERROR(data_manager.CopyTensor(values, sparse_tensor.MutableValues()));
    ORT_RETURN_IF_ERROR(data_manager.CopyTensor(inner_indices, rep->MutableInner()));
    ORT_RETURN_IF_ERROR(data_manager.CopyTensor(outer_indices, rep->MutableOuter()));

    dst = std::move(sparse_tensor);
    return Status::OK();
  }
};

common::Status DenseTensorToSparseCrcs(const DataTransferManager& data_manager, const Tensor& src,
                                       const AllocatorPtr& cpu_allocator, const AllocatorPtr& allocator,
                                       SparseTensor& dst) {
  const auto num_dims = src.Shape().NumDimensions();
  if (num_dims > 2) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Currently do not support dims higher than 2 dimensions");
  }

  utils::MLTypeCallDispatcher<int8_t, uint8_t, int16_t, uint16_t, int32_t, uint32_t,
                              int64_t, uint64_t, double, float, MLFloat16>
      t_disp(src.GetElementType());

  Status status;
  if (src.Location().device != cpu_allocator->Info().device) {
    Tensor src_cpu(src.DataType(), src.Shape(), cpu_allocator);
    ORT_RETURN_IF_ERROR(data_manager.CopyTensor(src, src_cpu));
    status = t_disp.InvokeRet<common::Status, ToSparseConvert>(data_manager, src_cpu, allocator, dst);
  } else {
    status = t_disp.InvokeRet<common::Status, ToSparseConvert>(data_manager, src, allocator, dst);
  }

  return status;
}
}  // namespace sparse_utils
}  // namespace onnxruntime