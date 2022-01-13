//===----------------------- Queue.h - RPC Queue ------------------*-c++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_UNITTESTS_EXECUTIONENGINE_ORC_QUEUECHANNEL_H
#define LLVM_UNITTESTS_EXECUTIONENGINE_ORC_QUEUECHANNEL_H

#include "llvm/ExecutionEngine/Orc/Shared/RawByteChannel.h"
#include "llvm/Support/Error.h"

#include <atomic>
#include <condition_variable>
#include <queue>

namespace llvm {

class QueueChannelError : public ErrorInfo<QueueChannelError> {
public:
  static char ID;
};

class QueueChannelClosedError
    : public ErrorInfo<QueueChannelClosedError, QueueChannelError> {
public:
  static char ID;
  std::error_code convertToErrorCode() const override {
    return inconvertibleErrorCode();
  }

  void log(raw_ostream &OS) const override {
    OS << "Queue closed";
  }
};

class Queue : public std::queue<char> {
public:
  using ErrorInjector = std::function<Error()>;

  Queue()
    : ReadError([]() { return Error::success(); }),
      WriteError([]() { return Error::success(); }) {}

  Queue(const Queue&) = delete;
  Queue& operator=(const Queue&) = delete;
  Queue(Queue&&) = delete;
  Queue& operator=(Queue&&) = delete;

  std::mutex &getMutex() { return M; }
  std::condition_variable &getCondVar() { return CV; }
  Error checkReadError() { return ReadError(); }
  Error checkWriteError() { return WriteError(); }
  void setReadError(ErrorInjector NewReadError) {
    {
      std::lock_guard<std::mutex> Lock(M);
      ReadError = std::move(NewReadError);
    }
    CV.notify_one();
  }
  void setWriteError(ErrorInjector NewWriteError) {
    std::lock_guard<std::mutex> Lock(M);
    WriteError = std::move(NewWriteError);
  }
private:
  std::mutex M;
  std::condition_variable CV;
  std::function<Error()> ReadError, WriteError;
};

class QueueChannel : public orc::shared::RawByteChannel {
public:
  QueueChannel(std::shared_ptr<Queue> InQueue,
               std::shared_ptr<Queue> OutQueue)
      : InQueue(InQueue), OutQueue(OutQueue) {}

  QueueChannel(const QueueChannel&) = delete;
  QueueChannel& operator=(const QueueChannel&) = delete;
  QueueChannel(QueueChannel&&) = delete;
  QueueChannel& operator=(QueueChannel&&) = delete;

  template <typename FunctionIdT, typename SequenceIdT>
  Error startSendMessage(const FunctionIdT &FnId, const SequenceIdT &SeqNo) {
    ++InFlightOutgoingMessages;
    return orc::shared::RawByteChannel::startSendMessage(FnId, SeqNo);
  }

  Error endSendMessage() {
    --InFlightOutgoingMessages;
    ++CompletedOutgoingMessages;
    return orc::shared::RawByteChannel::endSendMessage();
  }

  template <typename FunctionIdT, typename SequenceNumberT>
  Error startReceiveMessage(FunctionIdT &FnId, SequenceNumberT &SeqNo) {
    ++InFlightIncomingMessages;
    return orc::shared::RawByteChannel::startReceiveMessage(FnId, SeqNo);
  }

  Error endReceiveMessage() {
    --InFlightIncomingMessages;
    ++CompletedIncomingMessages;
    return orc::shared::RawByteChannel::endReceiveMessage();
  }

  Error readBytes(char *Dst, unsigned Size) override {
    std::unique_lock<std::mutex> Lock(InQueue->getMutex());
    while (Size) {
      {
        Error Err = InQueue->checkReadError();
        while (!Err && InQueue->empty()) {
          InQueue->getCondVar().wait(Lock);
          Err = InQueue->checkReadError();
        }
        if (Err)
          return Err;
      }
      *Dst++ = InQueue->front();
      --Size;
      ++NumRead;
      InQueue->pop();
    }
    return Error::success();
  }

  Error appendBytes(const char *Src, unsigned Size) override {
    std::unique_lock<std::mutex> Lock(OutQueue->getMutex());
    while (Size--) {
      if (Error Err = OutQueue->checkWriteError())
        return Err;
      OutQueue->push(*Src++);
      ++NumWritten;
    }
    OutQueue->getCondVar().notify_one();
    return Error::success();
  }

  Error send() override {
    ++SendCalls;
    return Error::success();
  }

  void close() {
    auto ChannelClosed = []() { return make_error<QueueChannelClosedError>(); };
    InQueue->setReadError(ChannelClosed);
    InQueue->setWriteError(ChannelClosed);
    OutQueue->setReadError(ChannelClosed);
    OutQueue->setWriteError(ChannelClosed);
  }

  uint64_t NumWritten = 0;
  uint64_t NumRead = 0;
  std::atomic<size_t> InFlightIncomingMessages{0};
  std::atomic<size_t> CompletedIncomingMessages{0};
  std::atomic<size_t> InFlightOutgoingMessages{0};
  std::atomic<size_t> CompletedOutgoingMessages{0};
  std::atomic<size_t> SendCalls{0};

private:

  std::shared_ptr<Queue> InQueue;
  std::shared_ptr<Queue> OutQueue;
};

inline std::pair<std::unique_ptr<QueueChannel>, std::unique_ptr<QueueChannel>>
createPairedQueueChannels() {
  auto Q1 = std::make_shared<Queue>();
  auto Q2 = std::make_shared<Queue>();
  auto C1 = std::make_unique<QueueChannel>(Q1, Q2);
  auto C2 = std::make_unique<QueueChannel>(Q2, Q1);
  return std::make_pair(std::move(C1), std::move(C2));
}

}

#endif
