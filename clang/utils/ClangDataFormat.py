"""lldb data formatters for clang classes.

Usage
--
import this file in your ~/.lldbinit by adding this line:

command script import /path/to/ClangDataFormat.py

After that, instead of getting this:

(lldb) p Tok.Loc
(clang::SourceLocation) $0 = {
  (unsigned int) ID = 123582
}

you'll get:

(lldb) p Tok.Loc
(clang::SourceLocation) $4 = "/usr/include/i386/_types.h:37:1" (offset: 123582, file, local)
"""

import lldb

def __lldb_init_module(debugger, internal_dict):
	debugger.HandleCommand("type summary add -F ClangDataFormat.SourceLocation_summary clang::SourceLocation")
	debugger.HandleCommand("type summary add -F ClangDataFormat.StringRef_summary llvm::StringRef")

def SourceLocation_summary(srcloc, internal_dict):
	return SourceLocation(srcloc).summary()

def StringRef_summary(strref, internal_dict):
	return StringRef(strref).summary()

class SourceLocation(object):
	def __init__(self, srcloc):
		self.srcloc = srcloc
		self.ID = srcloc.GetChildAtIndex(0).GetValueAsUnsigned()
	
	def offset(self):
		return getValueFromExpression(self.srcloc, ".getOffset()").GetValueAsUnsigned()

	def isInvalid(self):
		return self.ID == 0

	def isMacro(self):
		return getValueFromExpression(self.srcloc, ".isMacroID()").GetValueAsUnsigned()

	def isLocal(self, srcmgr_path):
		return lldb.frame.EvaluateExpression("(%s).isLocalSourceLocation(%s)" % (srcmgr_path, getExpressionPath(self.srcloc))).GetValueAsUnsigned()

	def getPrint(self, srcmgr_path):
		print_str = getValueFromExpression(self.srcloc, ".printToString(%s)" % srcmgr_path)
		return print_str.GetSummary()

	def summary(self):
		if self.isInvalid():
			return "<invalid loc>"
		desc = "(offset: %d, %s)" % (self.offset(), "macro" if self.isMacro() else "file")
		srcmgr_path = findObjectExpressionPath("clang::SourceManager", lldb.frame)
		if srcmgr_path:
			desc = "%s (offset: %d, %s, %s)" % (self.getPrint(srcmgr_path), self.offset(), "macro" if self.isMacro() else "file", "local" if self.isLocal(srcmgr_path) else "loaded")
		return desc

class StringRef(object):
	def __init__(self, strref):
		self.strref = strref
		self.Data_value = strref.GetChildAtIndex(0)
		self.Length = strref.GetChildAtIndex(1).GetValueAsUnsigned()

	def summary(self):
		if self.Length == 0:
			return '""'
		data = self.Data_value.GetPointeeData(0, self.Length)
		error = lldb.SBError()
		string = data.ReadRawData(error, 0, data.GetByteSize())
		if error.Fail():
			return None
		return '"%s"' % string


# Key is a (function address, type name) tuple, value is the expression path for
# an object with such a type name from inside that function.
FramePathMapCache = {}

def findObjectExpressionPath(typename, frame):
	func_addr = frame.GetFunction().GetStartAddress().GetFileAddress()
	key = (func_addr, typename)
	try:
		return FramePathMapCache[key]
	except KeyError:
		#print "CACHE MISS"
		path = None
		obj = findObject(typename, frame)
		if obj:
			path = getExpressionPath(obj)
		FramePathMapCache[key] = path
		return path

def findObject(typename, frame):
	def getTypename(value):
		# FIXME: lldb should provide something like getBaseType
		ty = value.GetType()
		if ty.IsPointerType() or ty.IsReferenceType():
			return ty.GetPointeeType().GetName()
		return ty.GetName()

	def searchForType(value, searched):
		tyname = getTypename(value)
		#print "SEARCH:", getExpressionPath(value), value.GetType().GetName()
		if tyname == typename:
			return value
		ty = value.GetType()
		if not (ty.IsPointerType() or
		        ty.IsReferenceType() or
				# FIXME: lldb should provide something like getCanonicalType
		        tyname.startswith("llvm::IntrusiveRefCntPtr<") or
		        tyname.startswith("llvm::OwningPtr<")):
			return None
		# FIXME: Hashing for SBTypes does not seem to work correctly, uses the typename instead,
		# and not the canonical one unfortunately.
		if tyname in searched:
			return None
		searched.add(tyname)
		for i in range(value.GetNumChildren()):
			child = value.GetChildAtIndex(i, 0, False)
			found = searchForType(child, searched)
			if found:
				return found

	searched = set()
	value_list = frame.GetVariables(True, True, True, True)
	for val in value_list:
		found = searchForType(val, searched)
		if found:
			return found if not found.TypeIsPointerType() else found.Dereference()

def getValueFromExpression(val, expr):
	return lldb.frame.EvaluateExpression(getExpressionPath(val) + expr)

def getExpressionPath(val):
	stream = lldb.SBStream()
	val.GetExpressionPath(stream)
	return stream.GetData()
