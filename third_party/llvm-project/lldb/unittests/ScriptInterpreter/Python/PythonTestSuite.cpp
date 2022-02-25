//===-- PythonTestSuite.cpp -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "gtest/gtest.h"

#include "Plugins/ScriptInterpreter/Python/lldb-python.h"

#include "Plugins/ScriptInterpreter/Python/ScriptInterpreterPython.h"
#include "Plugins/ScriptInterpreter/Python/ScriptInterpreterPythonImpl.h"
#include "lldb/API/SBError.h"
#include "lldb/Host/FileSystem.h"
#include "lldb/Host/HostInfo.h"

#include "PythonTestSuite.h"

using namespace lldb_private;
class TestScriptInterpreterPython : public ScriptInterpreterPythonImpl {
public:
  using ScriptInterpreterPythonImpl::Initialize;
  using ScriptInterpreterPythonImpl::InitializePrivate;
};

void PythonTestSuite::SetUp() {
  FileSystem::Initialize();
  HostInfoBase::Initialize();
  // ScriptInterpreterPython::Initialize() depends on HostInfo being
  // initializedso it can compute the python directory etc.
  TestScriptInterpreterPython::Initialize();
  TestScriptInterpreterPython::InitializePrivate();

  // Although we don't care about concurrency for the purposes of running
  // this test suite, Python requires the GIL to be locked even for
  // deallocating memory, which can happen when you call Py_DECREF or
  // Py_INCREF.  So acquire the GIL for the entire duration of this
  // test suite.
  m_gil_state = PyGILState_Ensure();
}

void PythonTestSuite::TearDown() {
  PyGILState_Release(m_gil_state);

  TestScriptInterpreterPython::Terminate();
  HostInfoBase::Terminate();
  FileSystem::Terminate();
}

// The following functions are the Pythonic implementations of the required
// callbacks. Because they're defined in libLLDB which we cannot link for the
// unit test, we have a 'default' implementation here.

#if PY_MAJOR_VERSION >= 3
extern "C" PyObject *PyInit__lldb(void) { return nullptr; }
#define LLDBSwigPyInit PyInit__lldb
#else
extern "C" void init_lldb(void) {}
#define LLDBSwigPyInit init_lldb
#endif

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wreturn-type-c-linkage"

// Disable warning C4190: 'LLDBSwigPythonBreakpointCallbackFunction' has
// C-linkage specified, but returns UDT 'llvm::Expected<bool>' which is
// incompatible with C
#if _MSC_VER
#pragma warning (push)
#pragma warning (disable : 4190)
#endif

extern "C" llvm::Expected<bool> LLDBSwigPythonBreakpointCallbackFunction(
    const char *python_function_name, const char *session_dictionary_name,
    const lldb::StackFrameSP &sb_frame,
    const lldb::BreakpointLocationSP &sb_bp_loc,
    StructuredDataImpl *args_impl) {
  return false;
}

#if _MSC_VER
#pragma warning (pop)
#endif

#pragma clang diagnostic pop

extern "C" bool LLDBSwigPythonWatchpointCallbackFunction(
    const char *python_function_name, const char *session_dictionary_name,
    const lldb::StackFrameSP &sb_frame, const lldb::WatchpointSP &sb_wp) {
  return false;
}

extern "C" bool LLDBSwigPythonCallTypeScript(
    const char *python_function_name, void *session_dictionary,
    const lldb::ValueObjectSP &valobj_sp, void **pyfunct_wrapper,
    const lldb::TypeSummaryOptionsSP &options_sp, std::string &retval) {
  return false;
}

extern "C" void *
LLDBSwigPythonCreateSyntheticProvider(const char *python_class_name,
                                      const char *session_dictionary_name,
                                      const lldb::ValueObjectSP &valobj_sp) {
  return nullptr;
}

extern "C" void *
LLDBSwigPythonCreateCommandObject(const char *python_class_name,
                                  const char *session_dictionary_name,
                                  const lldb::DebuggerSP debugger_sp) {
  return nullptr;
}

extern "C" void *LLDBSwigPythonCreateScriptedThreadPlan(
    const char *python_class_name, const char *session_dictionary_name,
    StructuredDataImpl *args_data,
    std::string &error_string,
    const lldb::ThreadPlanSP &thread_plan_sp) {
  return nullptr;
}

extern "C" bool LLDBSWIGPythonCallThreadPlan(void *implementor,
                                             const char *method_name,
                                             Event *event_sp, bool &got_error) {
  return false;
}

extern "C" void *LLDBSwigPythonCreateScriptedBreakpointResolver(
    const char *python_class_name, const char *session_dictionary_name,
    lldb_private::StructuredDataImpl *args, lldb::BreakpointSP &bkpt_sp) {
  return nullptr;
}

extern "C" unsigned int
LLDBSwigPythonCallBreakpointResolver(void *implementor, const char *method_name,
                                     lldb_private::SymbolContext *sym_ctx) {
  return 0;
}

extern "C" size_t LLDBSwigPython_CalculateNumChildren(void *implementor,
                                                      uint32_t max) {
  return 0;
}

extern "C" void *LLDBSwigPython_GetChildAtIndex(void *implementor,
                                                uint32_t idx) {
  return nullptr;
}

extern "C" int LLDBSwigPython_GetIndexOfChildWithName(void *implementor,
                                                      const char *child_name) {
  return 0;
}

extern "C" void *LLDBSWIGPython_CastPyObjectToSBData(void *data) {
  return nullptr;
}

extern "C" void *LLDBSWIGPython_CastPyObjectToSBError(void *data) {
  return nullptr;
}

extern "C" void *LLDBSWIGPython_CastPyObjectToSBValue(void *data) {
  return nullptr;
}

extern lldb::ValueObjectSP
LLDBSWIGPython_GetValueObjectSPFromSBValue(void *data) {
  return nullptr;
}

extern "C" bool LLDBSwigPython_UpdateSynthProviderInstance(void *implementor) {
  return false;
}

extern "C" bool
LLDBSwigPython_MightHaveChildrenSynthProviderInstance(void *implementor) {
  return false;
}

extern "C" void *
LLDBSwigPython_GetValueSynthProviderInstance(void *implementor) {
  return nullptr;
}

extern "C" bool
LLDBSwigPythonCallCommand(const char *python_function_name,
                          const char *session_dictionary_name,
                          lldb::DebuggerSP &debugger, const char *args,
                          lldb_private::CommandReturnObject &cmd_retobj,
                          lldb::ExecutionContextRefSP exe_ctx_ref_sp) {
  return false;
}

extern "C" bool
LLDBSwigPythonCallCommandObject(void *implementor, lldb::DebuggerSP &debugger,
                                const char *args,
                                lldb_private::CommandReturnObject &cmd_retobj,
                                lldb::ExecutionContextRefSP exe_ctx_ref_sp) {
  return false;
}

extern "C" bool
LLDBSwigPythonCallModuleInit(const char *python_module_name,
                             const char *session_dictionary_name,
                             lldb::DebuggerSP &debugger) {
  return false;
}

extern "C" void *
LLDBSWIGPythonCreateOSPlugin(const char *python_class_name,
                             const char *session_dictionary_name,
                             const lldb::ProcessSP &process_sp) {
  return nullptr;
}

extern "C" void *LLDBSwigPythonCreateScriptedProcess(
    const char *python_class_name, const char *session_dictionary_name,
    const lldb::TargetSP &target_sp, StructuredDataImpl *args_impl,
    std::string &error_string) {
  return nullptr;
}

extern "C" void *
LLDBSWIGPython_CreateFrameRecognizer(const char *python_class_name,
                                     const char *session_dictionary_name) {
  return nullptr;
}

extern "C" void *
LLDBSwigPython_GetRecognizedArguments(void *implementor,
                                      const lldb::StackFrameSP &frame_sp) {
  return nullptr;
}

extern "C" bool LLDBSWIGPythonRunScriptKeywordProcess(
    const char *python_function_name, const char *session_dictionary_name,
    lldb::ProcessSP &process, std::string &output) {
  return false;
}

extern "C" bool LLDBSWIGPythonRunScriptKeywordThread(
    const char *python_function_name, const char *session_dictionary_name,
    lldb::ThreadSP &thread, std::string &output) {
  return false;
}

extern "C" bool LLDBSWIGPythonRunScriptKeywordTarget(
    const char *python_function_name, const char *session_dictionary_name,
    lldb::TargetSP &target, std::string &output) {
  return false;
}

extern "C" bool LLDBSWIGPythonRunScriptKeywordFrame(
    const char *python_function_name, const char *session_dictionary_name,
    lldb::StackFrameSP &frame, std::string &output) {
  return false;
}

extern "C" bool LLDBSWIGPythonRunScriptKeywordValue(
    const char *python_function_name, const char *session_dictionary_name,
    lldb::ValueObjectSP &value, std::string &output) {
  return false;
}

extern "C" void *
LLDBSWIGPython_GetDynamicSetting(void *module, const char *setting,
                                 const lldb::TargetSP &target_sp) {
  return nullptr;
}

extern "C" void *LLDBSwigPythonCreateScriptedStopHook(
    lldb::TargetSP target_sp, const char *python_class_name,
    const char *session_dictionary_name,
    lldb_private::StructuredDataImpl *args_impl, Status &error) {
  return nullptr;
}

extern "C" bool
LLDBSwigPythonStopHookCallHandleStop(void *implementor,
                                     lldb::ExecutionContextRefSP exc_ctx_sp,
                                     lldb::StreamSP stream) {
  return false;
}
