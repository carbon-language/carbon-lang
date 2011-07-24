class fooSynthProvider:
     def __init__(self, valobj, dict):
         self.valobj = valobj;
     def num_children(self):
         return 2;
     def get_child_at_index(self, index):
         if index == 1:
             child = self.valobj.GetChildMemberWithName('a');
         else:
             child = self.valobj.GetChildMemberWithName('r');
         return child;
     def get_child_index(self, name):
         if name == 'a':
             return 1;
         else:
             return 0;