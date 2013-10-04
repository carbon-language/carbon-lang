//=== MipsMCAsmFlags.h - MipsMCAsmFlags --------------------------------===//
//
//                    The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENCE.TXT for details.
//
//===-------------------------------------------------------------------===//
#ifndef MIPSMCASMFLAGS_H_
#define MIPSMCASMFLAGS_H_

namespace llvm {
class MipsMCAsmFlags;

// We have the flags apart from the ELF defines because state will determine
// the final values put into the ELF flag bits.
//
// Currently we have only Relocation Model, but will soon follow with ABI,
// Architecture, and ASE.
class MipsMCAsmFlags {
public:
  // These act as bit flags because more that one can be
  // active at the same time, sometimes ;-)
  enum MAFRelocationModelTy {
    MAF_RM_DEFAULT = 0,
    MAF_RM_STATIC = 1,
    MAF_RM_CPIC = 2,
    MAF_RM_PIC = 4
  } MAFRelocationModel;

public:
  MipsMCAsmFlags() : Model(MAF_RM_DEFAULT) {}

  ~MipsMCAsmFlags() {}

  // Setting a bit we can later translate to the ELF header flags.
  void setRelocationModel(unsigned RM) { (Model |= RM); }

  bool isModelCpic() const { return (Model & MAF_RM_CPIC) == MAF_RM_CPIC; }
  bool isModelPic() const { return (Model & MAF_RM_PIC) == MAF_RM_PIC; }
  bool isModelStatic() const {
    return (Model & MAF_RM_STATIC) == MAF_RM_STATIC;
  }
  bool isModelDefault() const {
    return (Model & MAF_RM_DEFAULT) == MAF_RM_DEFAULT;
  }

private:
  unsigned Model; // pic, cpic, etc.
};
}

#endif /* MIPSMCASMFLAGS_H_ */
