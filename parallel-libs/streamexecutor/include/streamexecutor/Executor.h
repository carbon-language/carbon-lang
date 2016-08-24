//===-- Executor.h - The Executor class -------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// The Executor class which represents a single device of a specific platform.
///
//===----------------------------------------------------------------------===//

#ifndef STREAMEXECUTOR_EXECUTOR_H
#define STREAMEXECUTOR_EXECUTOR_H

#include "streamexecutor/KernelSpec.h"
#include "streamexecutor/PlatformInterfaces.h"
#include "streamexecutor/Utils/Error.h"

namespace streamexecutor {

class KernelInterface;
class Stream;

class Executor {
public:
  explicit Executor(PlatformExecutor *PExecutor);
  virtual ~Executor();

  /// Gets the kernel implementation for the underlying platform.
  virtual Expected<std::unique_ptr<KernelInterface>>
  getKernelImplementation(const MultiKernelLoaderSpec &Spec) {
    // TODO(jhen): Implement this.
    return nullptr;
  }

  Expected<std::unique_ptr<Stream>> createStream();

  /// Allocates an array of ElementCount entries of type T in device memory.
  template <typename T>
  Expected<GlobalDeviceMemory<T>> allocateDeviceMemory(size_t ElementCount) {
    return PExecutor->allocateDeviceMemory(ElementCount * sizeof(T));
  }

  /// Frees memory previously allocated with allocateDeviceMemory.
  template <typename T> Error freeDeviceMemory(GlobalDeviceMemory<T> Memory) {
    return PExecutor->freeDeviceMemory(Memory);
  }

  /// Allocates an array of ElementCount entries of type T in host memory.
  ///
  /// Host memory allocated by this function can be used for asynchronous memory
  /// copies on streams. See Stream::thenCopyD2H and Stream::thenCopyH2D.
  template <typename T> Expected<T *> allocateHostMemory(size_t ElementCount) {
    return PExecutor->allocateHostMemory(ElementCount * sizeof(T));
  }

  /// Frees memory previously allocated with allocateHostMemory.
  template <typename T> Error freeHostMemory(T *Memory) {
    return PExecutor->freeHostMemory(Memory);
  }

  /// Registers a previously allocated host array of type T for asynchronous
  /// memory operations.
  ///
  /// Host memory registered by this function can be used for asynchronous
  /// memory copies on streams. See Stream::thenCopyD2H and Stream::thenCopyH2D.
  template <typename T>
  Error registerHostMemory(T *Memory, size_t ElementCount) {
    return PExecutor->registerHostMemory(Memory, ElementCount * sizeof(T));
  }

  /// Unregisters host memory previously registered by registerHostMemory.
  template <typename T> Error unregisterHostMemory(T *Memory) {
    return PExecutor->unregisterHostMemory(Memory);
  }

  /// Host-synchronously copies a slice of an array of elements of type T from
  /// host to device memory.
  ///
  /// Returns an error if ElementCount is too large for the source slice or the
  /// destination.
  ///
  /// The calling host thread is blocked until the copy completes. Can be used
  /// with any host memory, the host memory does not have to be allocated with
  /// allocateHostMemory or registered with registerHostMemory. Does not block
  /// any ongoing device calls.
  template <typename T>
  Error synchronousCopyD2H(GlobalDeviceMemorySlice<T> Src,
                           llvm::MutableArrayRef<T> Dst, size_t ElementCount) {
    if (ElementCount > Src.getElementCount())
      return make_error("copying too many elements, " +
                        llvm::Twine(ElementCount) +
                        ", from a device array of element count " +
                        llvm::Twine(Src.getElementCount()));
    if (ElementCount > Dst.size())
      return make_error(
          "copying too many elements, " + llvm::Twine(ElementCount) +
          ", to a host array of element count " + llvm::Twine(Dst.size()));
    return PExecutor->synchronousCopyD2H(
        Src.getBaseMemory(), Src.getElementOffset() * sizeof(T), Dst.data(), 0,
        ElementCount * sizeof(T));
  }

  /// Similar to synchronousCopyD2H(GlobalDeviceMemorySlice<T>,
  /// llvm::MutableArrayRef<T>, size_t) but does not take an element count
  /// argument because it copies the entire source array.
  ///
  /// Returns an error if the Src and Dst sizes do not match.
  template <typename T>
  Error synchronousCopyD2H(GlobalDeviceMemorySlice<T> Src,
                           llvm::MutableArrayRef<T> Dst) {
    if (Src.getElementCount() != Dst.size())
      return make_error(
          "array size mismatch for D2H, device source has element count " +
          llvm::Twine(Src.getElementCount()) +
          " but host destination has element count " + llvm::Twine(Dst.size()));
    return synchronousCopyD2H(Src, Dst, Src.getElementCount());
  }

  /// Similar to synchronousCopyD2H(GlobalDeviceMemorySlice<T>,
  /// llvm::MutableArrayRef<T>, size_t) but copies to a pointer rather than an
  /// llvm::MutableArrayRef.
  ///
  /// Returns an error if ElementCount is too large for the source slice.
  template <typename T>
  Error synchronousCopyD2H(GlobalDeviceMemorySlice<T> Src, T *Dst,
                           size_t ElementCount) {
    return synchronousCopyD2H(Src, llvm::MutableArrayRef<T>(Dst, ElementCount),
                              ElementCount);
  }

  /// Similar to synchronousCopyD2H(GlobalDeviceMemorySlice<T>,
  /// llvm::MutableArrayRef<T>, size_t) but the source is a GlobalDeviceMemory
  /// rather than a GlobalDeviceMemorySlice.
  template <typename T>
  Error synchronousCopyD2H(GlobalDeviceMemory<T> Src,
                           llvm::MutableArrayRef<T> Dst, size_t ElementCount) {
    return synchronousCopyD2H(Src.asSlice(), Dst, ElementCount);
  }

  /// Similar to  synchronousCopyD2H(GlobalDeviceMemorySlice<T>,
  /// llvm::MutableArrayRef<T>) but the source is a GlobalDeviceMemory rather
  /// than a GlobalDeviceMemorySlice.
  template <typename T>
  Error synchronousCopyD2H(GlobalDeviceMemory<T> Src,
                           llvm::MutableArrayRef<T> Dst) {
    return synchronousCopyD2H(Src.asSlice(), Dst);
  }

  /// Similar to synchronousCopyD2H(GlobalDeviceMemorySlice<T>, T*, size_t) but
  /// the source is a GlobalDeviceMemory rather than a GlobalDeviceMemorySlice.
  template <typename T>
  Error synchronousCopyD2H(GlobalDeviceMemory<T> Src, T *Dst,
                           size_t ElementCount) {
    return synchronousCopyD2H(Src.asSlice(), Dst, ElementCount);
  }

  /// Host-synchronously copies a slice of an array of elements of type T from
  /// device to host memory.
  ///
  /// Returns an error if ElementCount is too large for the source or the
  /// destination.
  ///
  /// The calling host thread is blocked until the copy completes. Can be used
  /// with any host memory, the host memory does not have to be allocated with
  /// allocateHostMemory or registered with registerHostMemory. Does not block
  /// any ongoing device calls.
  template <typename T>
  Error synchronousCopyH2D(llvm::ArrayRef<T> Src,
                           GlobalDeviceMemorySlice<T> Dst,
                           size_t ElementCount) {
    if (ElementCount > Src.size())
      return make_error(
          "copying too many elements, " + llvm::Twine(ElementCount) +
          ", from a host array of element count " + llvm::Twine(Src.size()));
    if (ElementCount > Dst.getElementCount())
      return make_error("copying too many elements, " +
                        llvm::Twine(ElementCount) +
                        ", to a device array of element count " +
                        llvm::Twine(Dst.getElementCount()));
    return PExecutor->synchronousCopyH2D(Src.data(), 0, Dst.getBaseMemory(),
                                         Dst.getElementOffset() * sizeof(T),
                                         ElementCount * sizeof(T));
  }

  /// Similar to synchronousCopyH2D(llvm::ArrayRef<T>,
  /// GlobalDeviceMemorySlice<T>, size_t) but does not take an element count
  /// argument because it copies the entire source array.
  ///
  /// Returns an error if the Src and Dst sizes do not match.
  template <typename T>
  Error synchronousCopyH2D(llvm::ArrayRef<T> Src,
                           GlobalDeviceMemorySlice<T> Dst) {
    if (Src.size() != Dst.getElementCount())
      return make_error(
          "array size mismatch for H2D, host source has element count " +
          llvm::Twine(Src.size()) +
          " but device destination has element count " +
          llvm::Twine(Dst.getElementCount()));
    return synchronousCopyH2D(Src, Dst, Dst.getElementCount());
  }

  /// Similar to synchronousCopyH2D(llvm::ArrayRef<T>,
  /// GlobalDeviceMemorySlice<T>, size_t) but copies from a pointer rather than
  /// an llvm::ArrayRef.
  ///
  /// Returns an error if ElementCount is too large for the destination.
  template <typename T>
  Error synchronousCopyH2D(T *Src, GlobalDeviceMemorySlice<T> Dst,
                           size_t ElementCount) {
    return synchronousCopyH2D(llvm::ArrayRef<T>(Src, ElementCount), Dst,
                              ElementCount);
  }

  /// Similar to synchronousCopyH2D(llvm::ArrayRef<T>,
  /// GlobalDeviceMemorySlice<T>, size_t) but the destination is a
  /// GlobalDeviceMemory rather than a GlobalDeviceMemorySlice.
  template <typename T>
  Error synchronousCopyH2D(llvm::ArrayRef<T> Src, GlobalDeviceMemory<T> Dst,
                           size_t ElementCount) {
    return synchronousCopyH2D(Src, Dst.asSlice(), ElementCount);
  }

  /// Similar to synchronousCopyH2D(llvm::ArrayRef<T>,
  /// GlobalDeviceMemorySlice<T>) but the destination is a GlobalDeviceMemory
  /// rather than a GlobalDeviceMemorySlice.
  template <typename T>
  Error synchronousCopyH2D(llvm::ArrayRef<T> Src, GlobalDeviceMemory<T> Dst) {
    return synchronousCopyH2D(Src, Dst.asSlice());
  }

  /// Similar to synchronousCopyH2D(T*, GlobalDeviceMemorySlice<T>, size_t) but
  /// the destination is a GlobalDeviceMemory rather than a
  /// GlobalDeviceMemorySlice.
  template <typename T>
  Error synchronousCopyH2D(T *Src, GlobalDeviceMemory<T> Dst,
                           size_t ElementCount) {
    return synchronousCopyH2D(Src, Dst.asSlice(), ElementCount);
  }

  /// Host-synchronously copies a slice of an array of elements of type T from
  /// one location in device memory to another.
  ///
  /// Returns an error if ElementCount is too large for the source slice or the
  /// destination.
  ///
  /// The calling host thread is blocked until the copy completes. Can be used
  /// with any host memory, the host memory does not have to be allocated with
  /// allocateHostMemory or registered with registerHostMemory. Does not block
  /// any ongoing device calls.
  template <typename T>
  Error synchronousCopyD2D(GlobalDeviceMemorySlice<T> Src,
                           GlobalDeviceMemorySlice<T> Dst,
                           size_t ElementCount) {
    if (ElementCount > Src.getElementCount())
      return make_error("copying too many elements, " +
                        llvm::Twine(ElementCount) +
                        ", from a device array of element count " +
                        llvm::Twine(Src.getElementCount()));
    if (ElementCount > Dst.getElementCount())
      return make_error("copying too many elements, " +
                        llvm::Twine(ElementCount) +
                        ", to a device array of element count " +
                        llvm::Twine(Dst.getElementCount()));
    return PExecutor->synchronousCopyD2D(
        Src.getBaseMemory(), Src.getElementOffset() * sizeof(T),
        Dst.getBaseMemory(), Dst.getElementOffset() * sizeof(T),
        ElementCount * sizeof(T));
  }

  /// Similar to synchronousCopyD2D(GlobalDeviceMemorySlice<T>,
  /// GlobalDeviceMemorySlice<T>, size_t) but does not take an element count
  /// argument because it copies the entire source array.
  ///
  /// Returns an error if the Src and Dst sizes do not match.
  template <typename T>
  Error synchronousCopyD2D(GlobalDeviceMemorySlice<T> Src,
                           GlobalDeviceMemorySlice<T> Dst) {
    if (Src.getElementCount() != Dst.getElementCount())
      return make_error(
          "array size mismatch for D2D, device source has element count " +
          llvm::Twine(Src.getElementCount()) +
          " but device destination has element count " +
          llvm::Twine(Dst.getElementCount()));
    return synchronousCopyD2D(Src, Dst, Src.getElementCount());
  }

  /// Similar to synchronousCopyD2D(GlobalDeviceMemorySlice<T>,
  /// GlobalDeviceMemorySlice<T>, size_t) but the source is a
  /// GlobalDeviceMemory<T> rather than a GlobalDeviceMemorySlice<T>.
  template <typename T>
  Error synchronousCopyD2D(GlobalDeviceMemory<T> Src,
                           GlobalDeviceMemorySlice<T> Dst,
                           size_t ElementCount) {
    return synchronousCopyD2D(Src.asSlice(), Dst, ElementCount);
  }

  /// Similar to synchronousCopyD2D(GlobalDeviceMemorySlice<T>,
  /// GlobalDeviceMemorySlice<T>) but the source is a GlobalDeviceMemory<T>
  /// rather than a GlobalDeviceMemorySlice<T>.
  template <typename T>
  Error synchronousCopyD2D(GlobalDeviceMemory<T> Src,
                           GlobalDeviceMemorySlice<T> Dst) {
    return synchronousCopyD2D(Src.asSlice(), Dst);
  }

  /// Similar to synchronousCopyD2D(GlobalDeviceMemorySlice<T>,
  /// GlobalDeviceMemorySlice<T>, size_t) but the destination is a
  /// GlobalDeviceMemory<T> rather than a GlobalDeviceMemorySlice<T>.
  template <typename T>
  Error synchronousCopyD2D(GlobalDeviceMemorySlice<T> Src,
                           GlobalDeviceMemory<T> Dst, size_t ElementCount) {
    return synchronousCopyD2D(Src, Dst.asSlice(), ElementCount);
  }

  /// Similar to synchronousCopyD2D(GlobalDeviceMemorySlice<T>,
  /// GlobalDeviceMemorySlice<T>) but the destination is a GlobalDeviceMemory<T>
  /// rather than a GlobalDeviceMemorySlice<T>.
  template <typename T>
  Error synchronousCopyD2D(GlobalDeviceMemorySlice<T> Src,
                           GlobalDeviceMemory<T> Dst) {
    return synchronousCopyD2D(Src, Dst.asSlice());
  }

  /// Similar to synchronousCopyD2D(GlobalDeviceMemorySlice<T>,
  /// GlobalDeviceMemorySlice<T>, size_t) but the source and destination are
  /// GlobalDeviceMemory<T> rather than a GlobalDeviceMemorySlice<T>.
  template <typename T>
  Error synchronousCopyD2D(GlobalDeviceMemory<T> Src, GlobalDeviceMemory<T> Dst,
                           size_t ElementCount) {
    return synchronousCopyD2D(Src.asSlice(), Dst.asSlice(), ElementCount);
  }

  /// Similar to synchronousCopyD2D(GlobalDeviceMemorySlice<T>,
  /// GlobalDeviceMemorySlice<T>) but the source and destination are
  /// GlobalDeviceMemory<T> rather than a GlobalDeviceMemorySlice<T>.
  template <typename T>
  Error synchronousCopyD2D(GlobalDeviceMemory<T> Src,
                           GlobalDeviceMemory<T> Dst) {
    return synchronousCopyD2D(Src.asSlice(), Dst.asSlice());
  }

private:
  PlatformExecutor *PExecutor;
};

} // namespace streamexecutor

#endif // STREAMEXECUTOR_EXECUTOR_H
