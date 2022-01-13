//===- WriterUtils.h --------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLD_WASM_WRITERUTILS_H
#define LLD_WASM_WRITERUTILS_H

#include "lld/Common/LLVM.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Object/Wasm.h"

namespace lld {
namespace wasm {

void debugWrite(uint64_t offset, const Twine &msg);

void writeUleb128(raw_ostream &os, uint64_t number, const Twine &msg);

void writeSleb128(raw_ostream &os, int64_t number, const Twine &msg);

void writeBytes(raw_ostream &os, const char *bytes, size_t count,
                const Twine &msg);

void writeStr(raw_ostream &os, StringRef string, const Twine &msg);

void writeU8(raw_ostream &os, uint8_t byte, const Twine &msg);

void writeU32(raw_ostream &os, uint32_t number, const Twine &msg);

void writeValueType(raw_ostream &os, llvm::wasm::ValType type,
                    const Twine &msg);

void writeSig(raw_ostream &os, const llvm::wasm::WasmSignature &sig);

void writeI32Const(raw_ostream &os, int32_t number, const Twine &msg);

void writeI64Const(raw_ostream &os, int64_t number, const Twine &msg);

void writePtrConst(raw_ostream &os, int64_t number, bool is64,
                   const Twine &msg);

void writeMemArg(raw_ostream &os, uint32_t alignment, uint64_t offset);

void writeInitExpr(raw_ostream &os, const llvm::wasm::WasmInitExpr &initExpr);

void writeLimits(raw_ostream &os, const llvm::wasm::WasmLimits &limits);

void writeGlobalType(raw_ostream &os, const llvm::wasm::WasmGlobalType &type);

void writeTagType(raw_ostream &os, const llvm::wasm::WasmTagType &type);

void writeTag(raw_ostream &os, const llvm::wasm::WasmTag &tag);

void writeTableType(raw_ostream &os, const llvm::wasm::WasmTableType &type);

void writeImport(raw_ostream &os, const llvm::wasm::WasmImport &import);

void writeExport(raw_ostream &os, const llvm::wasm::WasmExport &export_);

} // namespace wasm

std::string toString(llvm::wasm::ValType type);
std::string toString(const llvm::wasm::WasmSignature &sig);
std::string toString(const llvm::wasm::WasmGlobalType &type);
std::string toString(const llvm::wasm::WasmTagType &type);
std::string toString(const llvm::wasm::WasmTableType &type);

} // namespace lld

#endif // LLD_WASM_WRITERUTILS_H
