"""Flexible enumeration of C types."""

from Enumeration import *

# TODO:

#  - struct improvements (bitfields, flexible arrays, packed &
#    unpacked, alignment)
#  - objective-c qualified id
#  - anonymous / transparent unions
#  - VLAs
#  - block types
#  - K&R functions
#  - pass arguments of different types (test extension, transparent union)
#  - varargs

###
# Actual type types

class BuiltinType:
    def __init__(self, name, size):
        self.name = name
        self.size = size

    def sizeof(self):
        return self.size

    def __str__(self):
        return self.name

class RecordType:
    def __init__(self, index, isUnion, fields):
        self.index = index
        self.isUnion = isUnion
        self.fields = fields
        self.name = None

    def __str__(self):        
        return '%s { %s }'%(('struct','union')[self.isUnion],
                            ' '.join(['%s;'%f for f in self.fields]))

    def getTypedefDef(self, name, printer):
        fields = ['%s field%d;'%(printer.getTypeName(t),i) for i,t in enumerate(self.fields)]
        # Name the struct for more readable LLVM IR.
        return 'typedef %s %s { %s } %s;'%(('struct','union')[self.isUnion],
                                           name, ' '.join(fields), name)
                                           
class ArrayType:
    def __init__(self, index, isVector, elementType, size):
        if isVector:
            # Note that for vectors, this is the size in bytes.
            assert size > 0
        else:
            assert size is None or size >= 0
        self.index = index
        self.isVector = isVector
        self.elementType = elementType
        self.size = size
        if isVector:
            eltSize = self.elementType.sizeof()
            assert not (self.size % eltSize)
            self.numElements = self.size // eltSize
        else:
            self.numElements = self.size

    def __str__(self):
        if self.isVector:
            return 'vector (%s)[%d]'%(self.elementType,self.size)
        elif self.size is not None:
            return '(%s)[%d]'%(self.elementType,self.size)
        else:
            return '(%s)[]'%(self.elementType,)

    def getTypedefDef(self, name, printer):
        elementName = printer.getTypeName(self.elementType)
        if self.isVector:
            return 'typedef %s %s __attribute__ ((vector_size (%d)));'%(elementName,
                                                                        name,
                                                                        self.size)
        else:
            if self.size is None:
                sizeStr = ''
            else:
                sizeStr = str(self.size)
            return 'typedef %s %s[%s];'%(elementName, name, sizeStr)

class ComplexType:
    def __init__(self, index, elementType):
        self.index = index
        self.elementType = elementType

    def __str__(self):
        return '_Complex (%s)'%(self.elementType)

    def getTypedefDef(self, name, printer):
        return 'typedef _Complex %s %s;'%(printer.getTypeName(self.elementType), name)

class FunctionType:
    def __init__(self, index, returnType, argTypes):
        self.index = index
        self.returnType = returnType
        self.argTypes = argTypes

    def __str__(self):
        if self.returnType is None:
            rt = 'void'
        else:
            rt = str(self.returnType)
        if not self.argTypes:
            at = 'void'
        else:
            at = ', '.join(map(str, self.argTypes))
        return '%s (*)(%s)'%(rt, at)

    def getTypedefDef(self, name, printer):
        if self.returnType is None:
            rt = 'void'
        else:
            rt = str(self.returnType)
        if not self.argTypes:
            at = 'void'
        else:
            at = ', '.join(map(str, self.argTypes))
        return 'typedef %s (*%s)(%s);'%(rt, name, at)

###
# Type enumerators

class TypeGenerator(object):
    def __init__(self):
        self.cache = {}

    def setCardinality(self):
        abstract

    def get(self, N):
        T = self.cache.get(N)
        if T is None:
            assert 0 <= N < self.cardinality
            T = self.cache[N] = self.generateType(N)
        return T

    def generateType(self, N):
        abstract

class FixedTypeGenerator(TypeGenerator):
    def __init__(self, types):
        TypeGenerator.__init__(self)
        self.types = types
        self.setCardinality()

    def setCardinality(self):
        self.cardinality = len(self.types)

    def generateType(self, N):
        return self.types[N]

class ComplexTypeGenerator(TypeGenerator):
    def __init__(self, typeGen):
        TypeGenerator.__init__(self)
        self.typeGen = typeGen
        self.setCardinality()
    
    def setCardinality(self):
        self.cardinality = self.typeGen.cardinality

    def generateType(self, N):
        return ComplexType(N, self.typeGen.get(N))

class VectorTypeGenerator(TypeGenerator):
    def __init__(self, typeGen, sizes):
        TypeGenerator.__init__(self)
        self.typeGen = typeGen
        self.sizes = tuple(map(int,sizes))
        self.setCardinality()

    def setCardinality(self):
        self.cardinality = len(self.sizes)*self.typeGen.cardinality

    def generateType(self, N):
        S,T = getNthPairBounded(N, len(self.sizes), self.typeGen.cardinality)
        return ArrayType(N, True, self.typeGen.get(T), self.sizes[S])

class FixedArrayTypeGenerator(TypeGenerator):
    def __init__(self, typeGen, sizes):
        TypeGenerator.__init__(self)
        self.typeGen = typeGen
        self.sizes = tuple(size)
        self.setCardinality()

    def setCardinality(self):
        self.cardinality = len(self.sizes)*self.typeGen.cardinality

    def generateType(self, N):
        S,T = getNthPairBounded(N, len(self.sizes), self.typeGen.cardinality)
        return ArrayType(N, false, self.typeGen.get(T), self.sizes[S])

class ArrayTypeGenerator(TypeGenerator):
    def __init__(self, typeGen, maxSize, useIncomplete=False, useZero=False):
        TypeGenerator.__init__(self)
        self.typeGen = typeGen
        self.useIncomplete = useIncomplete
        self.useZero = useZero
        self.maxSize = int(maxSize)
        self.W = useIncomplete + useZero + self.maxSize
        self.setCardinality()

    def setCardinality(self):
        self.cardinality = self.W * self.typeGen.cardinality

    def generateType(self, N):
        S,T = getNthPairBounded(N, self.W, self.typeGen.cardinality)
        if self.useIncomplete:
            if S==0:
                size = None
                S = None
            else:
                S = S - 1
        if S is not None:
            if self.useZero:
                size = S
            else:
                size = S + 1        
        return ArrayType(N, False, self.typeGen.get(T), size)

class RecordTypeGenerator(TypeGenerator):
    def __init__(self, typeGen, useUnion, maxSize):
        TypeGenerator.__init__(self)
        self.typeGen = typeGen
        self.useUnion = bool(useUnion)
        self.maxSize = int(maxSize)
        self.setCardinality()

    def setCardinality(self):
        M = 1 + self.useUnion
        if self.maxSize is aleph0:
            S =  aleph0 * self.typeGen.cardinality
        else:
            S = 0
            for i in range(self.maxSize+1):
                S += M * (self.typeGen.cardinality ** i)
        self.cardinality = S

    def generateType(self, N):
        isUnion,I = False,N
        if self.useUnion:
            isUnion,I = (I&1),I>>1
        fields = map(self.typeGen.get,getNthTuple(I,self.maxSize,self.typeGen.cardinality))
        return RecordType(N, isUnion, fields)

class FunctionTypeGenerator(TypeGenerator):
    def __init__(self, typeGen, useReturn, maxSize):
        TypeGenerator.__init__(self)
        self.typeGen = typeGen
        self.useReturn = useReturn
        self.maxSize = maxSize
        self.setCardinality()
    
    def setCardinality(self):
        if self.maxSize is aleph0:
            S = aleph0 * self.typeGen.cardinality()
        elif self.useReturn:
            S = 0
            for i in range(1,self.maxSize+1+1):
                S += self.typeGen.cardinality ** i
        else:
            S = 0
            for i in range(self.maxSize+1):
                S += self.typeGen.cardinality ** i
        self.cardinality = S
    
    def generateType(self, N):
        if self.useReturn:
            # Skip the empty tuple
            argIndices = getNthTuple(N+1, self.maxSize+1, self.typeGen.cardinality)
            retIndex,argIndices = argIndices[0],argIndices[1:]
            retTy = self.typeGen.get(retIndex)
        else:
            retTy = None
            argIndices = getNthTuple(N, self.maxSize, self.typeGen.cardinality)
        args = map(self.typeGen.get, argIndices)
        return FunctionType(N, retTy, args)

class AnyTypeGenerator(TypeGenerator):
    def __init__(self):
        TypeGenerator.__init__(self)
        self.generators = []
        self.bounds = []
        self.setCardinality()
        self._cardinality = None
        
    def getCardinality(self):
        if self._cardinality is None:
            return aleph0
        else:
            return self._cardinality
    def setCardinality(self):
        self.bounds = [g.cardinality for g in self.generators]
        self._cardinality = sum(self.bounds)
    cardinality = property(getCardinality, None)

    def addGenerator(self, g):
        self.generators.append(g)
        for i in range(100):
            prev = self._cardinality
            self._cardinality = None
            for g in self.generators:
                g.setCardinality()
            self.setCardinality()
            if (self._cardinality is aleph0) or prev==self._cardinality:
                break
        else:
            raise RuntimeError,"Infinite loop in setting cardinality"

    def generateType(self, N):
        index,M = getNthPairVariableBounds(N, self.bounds)
        return self.generators[index].get(M)

def test():
    atg = AnyTypeGenerator()
    btg = FixedTypeGenerator([BuiltinType('int',4),
                              BuiltinType('float',4)])
    atg.addGenerator( btg )
    atg.addGenerator( ComplexTypeGenerator(btg) )
    atg.addGenerator( RecordTypeGenerator(atg, True, 2) )
    atg.addGenerator( VectorTypeGenerator(btg, (4,8)) )
    atg.addGenerator( ArrayTypeGenerator(btg, 4) )
    atg.addGenerator( FunctionTypeGenerator(btg, False, 2) )
    print 'Cardinality:',atg.cardinality
    for i in range(100):
        print '%4d: %s'%(i, atg.get(i))

if __name__ == '__main__':
    test()
