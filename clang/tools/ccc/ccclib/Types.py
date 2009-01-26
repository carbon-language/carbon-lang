class InputType(object):
    """InputType - Information about various classes of files which
    the driver recognizes and control processing."""
    
    def __init__(self, name, preprocess=None, onlyAssemble=False, 
                 onlyPrecompile=False, tempSuffix=None, 
                 canBeUserSpecified=False):
        assert preprocess is None or isinstance(preprocess, InputType)
        self.name = name
        self.preprocess = preprocess
        self.onlyAssemble = onlyAssemble
        self.onlyPrecompile = onlyPrecompile
        self.tempSuffix = tempSuffix
        self.canBeUserSpecified = canBeUserSpecified

    def __repr__(self):
        return '%s(%r, %r, %r, %r, %r, %r)' % (self.__class__.__name__,
                                               self.name,
                                               self.preprocess, 
                                               self.onlyAssemble,
                                               self.onlyPrecompile,
                                               self.tempSuffix,
                                               self.canBeUserSpecified)

# C family source language (with and without preprocessing).
CTypeNoPP = InputType('cpp-output', tempSuffix='i', 
                      canBeUserSpecified=True)
CType = InputType('c', CTypeNoPP,
                  canBeUserSpecified=True)
ObjCTypeNoPP = InputType('objective-c-cpp-output', tempSuffix='mi',
                         canBeUserSpecified=True)
ObjCType = InputType('objective-c', ObjCTypeNoPP, 
                     canBeUserSpecified=True)
CXXTypeNoPP = InputType('c++-cpp-output', tempSuffix='ii',
                        canBeUserSpecified=True)
CXXType = InputType('c++', CXXTypeNoPP,
                    canBeUserSpecified=True)
ObjCXXTypeNoPP = InputType('objective-c++-cpp-output', tempSuffix='mii',
                           canBeUserSpecified=True)
ObjCXXType = InputType('objective-c++', ObjCXXTypeNoPP,
                       canBeUserSpecified=True)

# C family input files to precompile.
CHeaderNoPPType = InputType('c-header-cpp-output', tempSuffix='i',
                            onlyPrecompile=True)
CHeaderType = InputType('c-header', CHeaderNoPPType,
                        onlyPrecompile=True, canBeUserSpecified=True)
ObjCHeaderNoPPType = InputType('objective-c-header-cpp-output', tempSuffix='mi',
                               onlyPrecompile=True)
ObjCHeaderType = InputType('objective-c-header', ObjCHeaderNoPPType, 
                           onlyPrecompile=True, canBeUserSpecified=True)
CXXHeaderNoPPType = InputType('c++-header-cpp-output', tempSuffix='ii',
                              onlyPrecompile=True)
CXXHeaderType = InputType('c++-header', CXXHeaderNoPPType, 
                          onlyPrecompile=True, canBeUserSpecified=True)
ObjCXXHeaderNoPPType = InputType('objective-c++-header-cpp-output', tempSuffix='mii',
                                 onlyPrecompile=True)
ObjCXXHeaderType = InputType('objective-c++-header', ObjCXXHeaderNoPPType, 
                             onlyPrecompile=True, canBeUserSpecified=True)

# Other languages.
AdaType = InputType('ada', canBeUserSpecified=True)
AsmTypeNoPP = InputType('assembler', onlyAssemble=True, tempSuffix='s',
                        canBeUserSpecified=True)
AsmType = InputType('assembler-with-cpp', AsmTypeNoPP, onlyAssemble=True,
                    canBeUserSpecified=True)
FortranTypeNoPP = InputType('f95', canBeUserSpecified=True)
FortranType = InputType('f95-cpp-input', FortranTypeNoPP, canBeUserSpecified=True)
JavaType = InputType('java', canBeUserSpecified=True)

# Misc.
LLVMAsmType = InputType('llvm-asm', tempSuffix='ll')
LLVMBCType = InputType('llvm-bc', tempSuffix='bc')
PlistType = InputType('plist', tempSuffix='plist')
PCHType = InputType('precompiled-header', tempSuffix='gch')
ObjectType = InputType('object', tempSuffix='o')
TreelangType = InputType('treelang', canBeUserSpecified=True)
ImageType = InputType('image', tempSuffix='out')
NothingType = InputType('nothing')

###

kDefaultOutput = "a.out"
kTypeSuffixMap = {
    '.c' : CType,
    '.i' : CTypeNoPP,
    '.ii' : CXXTypeNoPP,
    '.m' : ObjCType,
    '.mi' : ObjCTypeNoPP,
    '.mm' : ObjCXXType,
    '.M' : ObjCXXType,
    '.mii' : ObjCXXTypeNoPP,
    '.h' : CHeaderType,
    '.cc' : CXXType,
    '.cc' : CXXType,
    '.cp' : CXXType,
    '.cxx' : CXXType,
    '.cpp' : CXXType,
    '.CPP' : CXXType,
    '.cXX' : CXXType,
    '.C' : CXXType,
    '.hh' : CXXHeaderType,
    '.H' : CXXHeaderType,
    '.f' : FortranTypeNoPP,
    '.for' : FortranTypeNoPP,
    '.FOR' : FortranTypeNoPP,
    '.F' : FortranType,
    '.fpp' : FortranType,
    '.FPP' : FortranType,
    '.f90' : FortranTypeNoPP,
    '.f95' : FortranTypeNoPP,
    '.F90' : FortranType,
    '.F95' : FortranType,
    # Apparently the Ada F-E hardcodes these suffixes in many
    # places. This explains why there is only one -x option for ada.
    '.ads' : AdaType,
    '.adb' : AdaType,
    # FIXME: Darwin always uses a preprocessor for asm input. Where
    # does this fit?
    '.s' : AsmTypeNoPP,
    '.S' : AsmType,
}
kTypeSpecifierMap = {
    'none' : None,

    'c' : CType,
    'c-header' : CHeaderType,
    # NOTE: gcc.info claims c-cpp-output works but the actual spelling
    # is cpp-output. Nice.
    'cpp-output' : CTypeNoPP,
    'c++' : CXXType, 
    'c++-header' : CXXHeaderType,
    'c++-cpp-output' : CXXTypeNoPP,
    'objective-c' : ObjCType,
    'objective-c-header' : ObjCHeaderType,
    'objective-c-cpp-output' : ObjCTypeNoPP,
    'objective-c++' : ObjCXXType,
    'objective-c++-header' : ObjCXXHeaderType,
    'objective-c++-cpp-output' : ObjCXXTypeNoPP,
    'assembler' : AsmTypeNoPP,
    'assembler-with-cpp' : AsmType,
    'ada' : AdaType,
    'f95-cpp-input' : FortranType, 
    'f95' : FortranTypeNoPP,
    'java' : JavaType,
    'treelang' : TreelangType,
}

# Check that the type specifier map at least matches what the types
# believe to be true.
assert not [name for name,type in kTypeSpecifierMap.items()
            if type and (type.name != name or not type.canBeUserSpecified)]
