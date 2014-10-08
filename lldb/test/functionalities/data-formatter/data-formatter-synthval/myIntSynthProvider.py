class myIntSynthProvider(object):
	def __init__(self, valobj, dict):
		self.valobj = valobj;
		self.val = self.valobj.GetChildMemberWithName("theValue")
	def num_children(self):
		return 0;
	def get_child_at_index(self, index):
	    return None
	def get_child_index(self, name):
	    return None
	def update(self):
		return False
	def might_have_children(self):
	    return False
	def get_value(self):
	    return self.val

