//===------ dxcontainer2yaml.cpp - obj2yaml conversion tool -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "obj2yaml.h"
#include "llvm/Object/DXContainer.h"
#include "llvm/ObjectYAML/DXContainerYAML.h"
#include "llvm/Support/Error.h"

#include <algorithm>

using namespace llvm;
using namespace llvm::object;

static Expected<DXContainerYAML::Object *>
dumpDXContainer(MemoryBufferRef Source) {
  assert(file_magic::dxcontainer_object == identify_magic(Source.getBuffer()));

  Expected<DXContainer> ExDXC = DXContainer::create(Source);
  if (!ExDXC)
    return ExDXC.takeError();
  DXContainer Container = *ExDXC;

  std::unique_ptr<DXContainerYAML::Object> Obj =
      std::make_unique<DXContainerYAML::Object>();

  for (uint8_t Byte : Container.getHeader().FileHash.Digest)
    Obj->Header.Hash.push_back(Byte);
  Obj->Header.Version.Major = Container.getHeader().Version.Major;
  Obj->Header.Version.Minor = Container.getHeader().Version.Minor;
  Obj->Header.FileSize = Container.getHeader().FileSize;
  Obj->Header.PartCount = Container.getHeader().PartCount;

  Obj->Header.PartOffsets = std::vector<uint32_t>();
  for (const auto P : Container) {
    Obj->Header.PartOffsets->push_back(P.Offset);
    if (P.Part.getName() == "DXIL") {
      Optional<DXContainer::DXILData> DXIL = Container.getDXIL();
      assert(DXIL.hasValue() && "Since we are iterating and found a DXIL part, "
                                "this should never not have a value");
      Obj->Parts.push_back(DXContainerYAML::Part{
          P.Part.getName().str(), P.Part.Size,
          DXContainerYAML::DXILProgram{
              DXIL->first.MajorVersion, DXIL->first.MinorVersion,
              DXIL->first.ShaderKind, DXIL->first.Size,
              DXIL->first.Bitcode.MajorVersion,
              DXIL->first.Bitcode.MinorVersion, DXIL->first.Bitcode.Offset,
              DXIL->first.Bitcode.Size,
              std::vector<llvm::yaml::Hex8>(
                  DXIL->second, DXIL->second + DXIL->first.Bitcode.Size)}});
    } else {
      Obj->Parts.push_back(
          DXContainerYAML::Part{P.Part.getName().str(), P.Part.Size, None});
    }
  }

  return Obj.release();
}

llvm::Error dxcontainer2yaml(llvm::raw_ostream &Out,
                             llvm::MemoryBufferRef Source) {
  Expected<DXContainerYAML::Object *> YAMLOrErr = dumpDXContainer(Source);
  if (!YAMLOrErr)
    return YAMLOrErr.takeError();

  std::unique_ptr<DXContainerYAML::Object> YAML(YAMLOrErr.get());
  yaml::Output Yout(Out);
  Yout << *YAML;

  return Error::success();
}
