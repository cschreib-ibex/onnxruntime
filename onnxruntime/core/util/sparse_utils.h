// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/allocator.h"

namespace onnxruntime {
class Tensor;
class SparseTensor;
class DataTransferManager;
namespace common {
class Status;
}

namespace sparse_utils {
common::Status DenseTensorToSparseCrcs(const DataTransferManager& data_manager, const Tensor& src, const AllocatorPtr& cpu_allocator, 
  const AllocatorPtr& src_allocator, SparseTensor& dst);
}
}
