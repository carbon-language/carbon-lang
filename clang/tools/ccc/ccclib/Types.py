class InputType(object):
    """InputType - Information about various classes of files which
    the driver recognizes and control processing."""
    
    def __init__(self, name, preprocess=None, onlyAssemble=False, 
                 onlyPrecompile=False, tempSuffix=None):
        assert preprocess is None or isinstance(preprocess, InputType)
        self.name = name
        self.preprocess = preprocess
        self.onlyAssemble = onlyAssemble
        self.onlyPrecompile = onlyPrecompile
        self.tempSuffix = tempSuffix

    def __repr__(self):
        return '%s(%r, %r, %r, %r, %r)' % (self.__class__.__name__,
                                           self.name,
                                           self.preprocess, 
                                           self.onlyAssemble,
                                           self.onlyPrecompile,
                                           self.tempSuffix)

# C family source language (with and without preprocessing).
CTypeNoPP = InputType('cpp-output', tempSuffix='i')
CType = InputType('c', CTypeNoPP)
ObjCTypeNoPP = InputType('objective-c-cpp-output', tempSuffix='mi')
ObjCType = InputType('objective-c', ObjCTypeNoPP)
CXXTypeNoPP = InputType('c++-cpp-output', tempSuffix='ii')
CXXType = InputType('c++', CXXTypeNoPP)
ObjCXXTypeNoPP = InputType('objective-c++-cpp-output', tempSuffix='mii')
ObjCXXType = InputType('c++', ObjCXXTypeNoPP)

# C family input files to precompile.
CHeaderNoPPType = InputType('c-header-cpp-output', onlyPrecompile=True, tempSuffix='pch')
CHeaderType = InputType('c-header', CHeaderNoPPType, onlyPrecompile=True)
ObjCHeaderNoPPType = InputType('objective-c-header-cpp-output', onlyPrecompile=True, tempSuffix='pch')
ObjCHeaderType = InputType('objective-c-header', ObjCHeaderNoPPType, onlyPrecompile=True)
CXXHeaderNoPPType = InputType('c++-header-cpp-output', onlyPrecompile=True, tempSuffix='pch')
CXXHeaderType = InputType('c++-header', CXXHeaderNoPPType, onlyPrecompile=True)
ObjCXXHeaderNoPPType = InputType('objective-c++-header-cpp-output', onlyPrecompile=True, tempSuffix='pch')
ObjCXXHeaderType = InputType('objective-c++-header', ObjCXXHeaderNoPPType, onlyPrecompile=True)

# Other languages.
AdaType = InputType('ada')
AsmTypeNoPP = InputType('assembler', onlyAssemble=True, tempSuffix='s')
AsmType = InputType('assembler-with-cpp', AsmTypeNoPP, onlyAssemble=True)
FortranTypeNoPP = InputType('fortran')
FortranType = InputType('fortran', FortranTypeNoPP)
JavaType = InputType('java')

# Misc.
PCHType = InputType('precompiled-header')
ObjectType = InputType('object', tempSuffix='o')
TreelangType = InputType('treelang')
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
    'f95' : FortranType, 
    'f95-cpp-input' : FortranTypeNoPP,
    'java' : JavaType,
    'treelang' : TreelangType,
}
