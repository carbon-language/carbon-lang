"""
LLDB AppKit formatters

part of The LLVM Compiler Infrastructure
This file is distributed under the University of Illinois Open Source
License. See LICENSE.TXT for details.
"""
import lldb
import objc_runtime
import Logger

def Class_Summary(valobj,dict):
	logger = Logger.Logger()
	runtime = objc_runtime.ObjCRuntime.runtime_from_isa(valobj)
	if runtime == None or not runtime.is_valid():
		return '<error: unknown Class>'
	class_data = runtime.read_class_data()
	if class_data == None or not class_data.is_valid():
		return '<error: unknown Class>'
	return class_data.class_name()

