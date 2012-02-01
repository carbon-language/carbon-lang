
%extend lldb::SBAddress {
        PyObject *lldb::SBAddress::__repr__ (){
                lldb::SBStream description;
                $self->GetDescription (description);
                return PyString_FromString (description.GetData());
        }
}
%extend lldb::SBBlock {
        PyObject *lldb::SBBlock::__repr__ (){
                lldb::SBStream description;
                $self->GetDescription (description);
                return PyString_FromString (description.GetData());
        }
}
%extend lldb::SBBreakpoint {
        PyObject *lldb::SBBreakpoint::__repr__ (){
                lldb::SBStream description;
                $self->GetDescription (description);
                return PyString_FromString (description.GetData());
        }
}
%extend lldb::SBBreakpointLocation {
        PyObject *lldb::SBBreakpointLocation::__repr__ (){
                lldb::SBStream description;
                $self->GetDescription (description, lldb::eDescriptionLevelFull);
                return PyString_FromString (description.GetData());
        }
}
%extend lldb::SBCommandReturnObject {
        PyObject *lldb::SBCommandReturnObject::__repr__ (){
                lldb::SBStream description;
                $self->GetDescription (description);
                return PyString_FromString (description.GetData());
        }
}
%extend lldb::SBCompileUnit {
        PyObject *lldb::SBCompileUnit::__repr__ (){
                lldb::SBStream description;
                $self->GetDescription (description);
                return PyString_FromString (description.GetData());
        }
}
%extend lldb::SBData {
        PyObject *lldb::SBData::__repr__ (){
                lldb::SBStream description;
                $self->GetDescription (description);
                return PyString_FromString (description.GetData());
        }
}
%extend lldb::SBDebugger {
        PyObject *lldb::SBDebugger::__repr__ (){
                lldb::SBStream description;
                $self->GetDescription (description);
                return PyString_FromString (description.GetData());
        }
}
%extend lldb::SBError {
        PyObject *lldb::SBError::__repr__ (){
                lldb::SBStream description;
                $self->GetDescription (description);
                return PyString_FromString (description.GetData());
        }
}
%extend lldb::SBFileSpec {
        PyObject *lldb::SBFileSpec::__repr__ (){
                lldb::SBStream description;
                $self->GetDescription (description);
                return PyString_FromString (description.GetData());
        }
}
%extend lldb::SBFrame {
        PyObject *lldb::SBFrame::__repr__ (){
                lldb::SBStream description;
                $self->GetDescription (description);
                return PyString_FromString (description.GetData());
        }
}
%extend lldb::SBFunction {
        PyObject *lldb::SBFunction::__repr__ (){
                lldb::SBStream description;
                $self->GetDescription (description);
                return PyString_FromString (description.GetData());
        }
}
%extend lldb::SBInstruction {
        PyObject *lldb::SBInstruction::__repr__ (){
                lldb::SBStream description;
                $self->GetDescription (description);
                return PyString_FromString (description.GetData());
        }
}
%extend lldb::SBInstructionList {
        PyObject *lldb::SBInstructionList::__repr__ (){
                lldb::SBStream description;
                $self->GetDescription (description);
                return PyString_FromString (description.GetData());
        }
}
%extend lldb::SBLineEntry {
        PyObject *lldb::SBLineEntry::__repr__ (){
                lldb::SBStream description;
                $self->GetDescription (description);
                return PyString_FromString (description.GetData());
        }
}
%extend lldb::SBModule {
        PyObject *lldb::SBModule::__repr__ (){
                lldb::SBStream description;
                $self->GetDescription (description);
                return PyString_FromString (description.GetData());
        }
}
%extend lldb::SBProcess {
        PyObject *lldb::SBProcess::__repr__ (){
                lldb::SBStream description;
                $self->GetDescription (description);
                return PyString_FromString (description.GetData());
        }
}
%extend lldb::SBSection {
        PyObject *lldb::SBSection::__repr__ (){
                lldb::SBStream description;
                $self->GetDescription (description);
                return PyString_FromString (description.GetData());
        }
}
%extend lldb::SBSymbol {
        PyObject *lldb::SBSymbol::__repr__ (){
                lldb::SBStream description;
                $self->GetDescription (description);
                return PyString_FromString (description.GetData());
        }
}
%extend lldb::SBSymbolContext {
        PyObject *lldb::SBSymbolContext::__repr__ (){
                lldb::SBStream description;
                $self->GetDescription (description);
                return PyString_FromString (description.GetData());
        }
}
%extend lldb::SBTarget {
        PyObject *lldb::SBTarget::__repr__ (){
                lldb::SBStream description;
                $self->GetDescription (description, lldb::eDescriptionLevelBrief);
                return PyString_FromString (description.GetData());
        }
}
%extend lldb::SBType {
        PyObject *lldb::SBType::__repr__ (){
                lldb::SBStream description;
                $self->GetDescription (description, lldb::eDescriptionLevelBrief);
                return PyString_FromString (description.GetData());
        }
}
%extend lldb::SBTypeMember {
        PyObject *lldb::SBTypeMember::__repr__ (){
                lldb::SBStream description;
                $self->GetDescription (description, lldb::eDescriptionLevelBrief);
                return PyString_FromString (description.GetData());
        }
}
%extend lldb::SBThread {
        PyObject *lldb::SBThread::__repr__ (){
                lldb::SBStream description;
                $self->GetDescription (description);
                return PyString_FromString (description.GetData());
        }
}
%extend lldb::SBValue {
        PyObject *lldb::SBValue::__repr__ (){
                lldb::SBStream description;
                $self->GetDescription (description);
                return PyString_FromString (description.GetData());
        }
}
%extend lldb::SBWatchpoint {
        PyObject *lldb::SBWatchpoint::__repr__ (){
                lldb::SBStream description;
                $self->GetDescription (description, lldb::eDescriptionLevelVerbose);
                return PyString_FromString (description.GetData());
        }
}

%pythoncode %{

class value(object):
    '''A class designed to wrap lldb.SBValue() objects so the resulting object
    can be used as a variable would be in code. So if you have a Point structure
    variable in your code in the current frame named "pt", you can initialize an instance
    of this class with it:
    
    pt = lldb.value(lldb.frame.FindVariable("pt"))
    print pt
    print pt.x
    print pt.y

    pt = lldb.value(lldb.frame.FindVariable("rectangle_array"))
    print rectangle_array[12]
    print rectangle_array[5].origin.x'''
    def __init__(self, sbvalue):
        self.sbvalue = sbvalue

    def __nonzero__(self):
        return self.sbvalue.__nonzero__()

    def __repr__(self):
        return self.sbvalue.__repr__()

    def __str__(self):
        return self.sbvalue.__str__()

    def __getitem__(self, key):
        # Allow array access if this value has children...
        if type(key) is int:
            return value(self.sbvalue.GetValueForExpressionPath("[%i]" % key))
        raise TypeError

    def __getattr__(self, name):
        child_sbvalue = self.sbvalue.GetChildMemberWithName (name)
        if child_sbvalue:
            return value(child_sbvalue)
        raise AttributeError

    def __add__(self, other):
        return int(self) + int(other)
        
    def __sub__(self, other):
        return int(self) - int(other)
        
    def __mul__(self, other):
        return int(self) * int(other)
        
    def __floordiv__(self, other):
        return int(self) // int(other)
        
    def __mod__(self, other):
        return int(self) % int(other)
        
    def __divmod__(self, other):
        return int(self) % int(other)
        
    def __pow__(self, other):
        return int(self) ** int(other)
        
    def __lshift__(self, other):
        return int(self) << int(other)
        
    def __rshift__(self, other):
        return int(self) >> int(other)
        
    def __and__(self, other):
        return int(self) & int(other)
        
    def __xor__(self, other):
        return int(self) ^ int(other)
        
    def __or__(self, other):
        return int(self) | int(other)
        
    def __div__(self, other):
        return int(self) / int(other)
        
    def __truediv__(self, other):
        return int(self) / int(other)
        
    def __iadd__(self, other):
        result = self.__add__(other)
        self.sbvalue.SetValueFromCString (str(result))
        return result
        
    def __isub__(self, other):
        result = self.__sub__(other)
        self.sbvalue.SetValueFromCString (str(result))
        return result
        
    def __imul__(self, other):
        result = self.__mul__(other)
        self.sbvalue.SetValueFromCString (str(result))
        return result
        
    def __idiv__(self, other):
        result = self.__div__(other)
        self.sbvalue.SetValueFromCString (str(result))
        return result
        
    def __itruediv__(self, other):
        result = self.__truediv__(other)
        self.sbvalue.SetValueFromCString (str(result))
        return result
        
    def __ifloordiv__(self, other):
        result =  self.__floordiv__(self, other)
        self.sbvalue.SetValueFromCString (str(result))
        return result
        
    def __imod__(self, other):
        result =  self.__and__(self, other)
        self.sbvalue.SetValueFromCString (str(result))
        return result
        
    def __ipow__(self, other):
        result = self.__pow__(self, other)
        self.sbvalue.SetValueFromCString (str(result))
        return result
        
    def __ipow__(self, other, modulo):
        result = self.__pow__(self, other, modulo)
        self.sbvalue.SetValueFromCString (str(result))
        return result
        
    def __ilshift__(self, other):
        result = self.__lshift__(other)
        self.sbvalue.SetValueFromCString (str(result))
        return result
        
    def __irshift__(self, other):
        result =  self.__rshift__(other)
        self.sbvalue.SetValueFromCString (str(result))
        return result
        
    def __iand__(self, other):
        result =  self.__and__(self, other)
        self.sbvalue.SetValueFromCString (str(result))
        return result
        
    def __ixor__(self, other):
        result =  self.__xor__(self, other)
        self.sbvalue.SetValueFromCString (str(result))
        return result
        
    def __ior__(self, other):
        result =  self.__ior__(self, other)
        self.sbvalue.SetValueFromCString (str(result))
        return result
        
    def __neg__(self):
        return -int(self)
        
    def __pos__(self):
        return +int(self)
        
    def __abs__(self):
        return abs(int(self))
        
    def __invert__(self):
        return ~int(self)
        
    def __complex__(self):
        return complex (int(self))
        
    def __int__(self):
        return self.sbvalue.GetValueAsSigned()
        
    def __long__(self):
        return self.sbvalue.GetValueAsSigned()
        
    def __float__(self):
        return float (self.sbvalue.GetValueAsSigned())
        
    def __oct__(self):
        return '0%o' % self.sbvalue.GetValueAsSigned()
        
    def __hex__(self):
        return '0x%x' % self.sbvalue.GetValueAsSigned()

%}

