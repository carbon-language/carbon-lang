# Summaries for common ObjC types that require Python scripting
# to be generated fit into this file

def BOOL_SummaryProvider (valobj,dict):
	if valobj.GetValueAsUnsigned() == 0:
		return "NO"
	else:
		return "YES"
		
