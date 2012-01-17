"""
Load into LLDB with:
script import lldbDataFormatters
type synthetic add -x "^llvm::SmallVectorImpl<.+>$" -l lldbDataFormatters.SmallVectorSynthProvider
"""

# Pretty printer for llvm::SmallVector/llvm::SmallVectorImpl
class SmallVectorSynthProvider:
    def __init__(self, valobj, dict):
        self.valobj = valobj;
        self.update() # initialize this provider

    def num_children(self):
        begin = self.begin.GetValueAsUnsigned(0)
        end = self.end.GetValueAsUnsigned(0)
        return (end - begin)/self.type_size

    def get_child_index(self, name):
        try:
            return int(name.lstrip('[').rstrip(']'))
        except:
            return -1;

    def get_child_at_index(self, index):
        # Do bounds checking.
        if index < 0:
            return None
        if index >= self.num_children():
            return None;

        offset = index * self.type_size
        return self.begin.CreateChildAtOffset('['+str(index)+']',
                                              offset, self.data_type)

    def get_type_from_name(self):
        import re
        name = self.valobj.GetType().GetName()
        # This class works with both SmallVectors and SmallVectorImpls.
        res = re.match("^(llvm::)?SmallVectorImpl<(.+)>$", name)
        if res:
            return res.group(2)
        res = re.match("^(llvm::)?SmallVector<(.+), \d+>$", name)
        if res:
            return res.group(2)
        return None

    def update(self):
        self.begin = self.valobj.GetChildMemberWithName('BeginX')
        self.end = self.valobj.GetChildMemberWithName('EndX')
        data_type = self.get_type_from_name()
        # FIXME: this sometimes returns an invalid type.
        self.data_type = self.valobj.GetTarget().FindFirstType(data_type)
        self.type_size = self.data_type.GetByteSize()
