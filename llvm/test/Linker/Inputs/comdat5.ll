target datalayout = "e-m:w-p:32:32-i64:64-f80:32-n8:16:32-S32"
target triple = "i686-pc-windows-msvc"

%MSRTTICompleteObjectLocator = type { i32, i32, i32, i8*, %MSRTTIClassHierarchyDescriptor* }
%MSRTTIClassHierarchyDescriptor = type { i32, i32, i32, %MSRTTIBaseClassDescriptor** }
%MSRTTIBaseClassDescriptor = type { i8*, i32, i32, i32, i32, i32, %MSRTTIClassHierarchyDescriptor* }
%struct.S = type { i32 (...)** }

$"\01??_7S@@6B@" = comdat largest

@"\01??_R4S@@6B@" = external constant %MSRTTICompleteObjectLocator
@some_name = private unnamed_addr constant [2 x i8*] [i8* bitcast (%MSRTTICompleteObjectLocator* @"\01??_R4S@@6B@" to i8*), i8* bitcast (void (%struct.S*, i32)* @"\01??_GS@@UAEPAXI@Z" to i8*)], comdat $"\01??_7S@@6B@"
@"\01??_7S@@6B@" = alias getelementptr([2 x i8*]* @some_name, i32 0, i32 1)

declare x86_thiscallcc void @"\01??_GS@@UAEPAXI@Z"(%struct.S*, i32) unnamed_addr
