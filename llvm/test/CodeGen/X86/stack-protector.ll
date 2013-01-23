; RUN: llc -mtriple=i386-pc-linux-gnu < %s -o - | FileCheck --check-prefix=LINUX-I386 %s
; RUN: llc -mtriple=x86_64-pc-linux-gnu < %s -o - | FileCheck --check-prefix=LINUX-X64 %s
; RUN: llc -code-model=kernel -mtriple=x86_64-pc-linux-gnu < %s -o - | FileCheck --check-prefix=LINUX-KERNEL-X64 %s
; RUN: llc -mtriple=x86_64-apple-darwin < %s -o - | FileCheck --check-prefix=DARWIN-X64 %s

%struct.foo = type { [16 x i8] }
%struct.foo.0 = type { [4 x i8] }
%struct.pair = type { i32, i32 }
%struct.nest = type { %struct.pair, %struct.pair }
%struct.vec = type { <4 x i32> }
%class.A = type { [2 x i8] }
%struct.deep = type { %union.anon }
%union.anon = type { %struct.anon }
%struct.anon = type { %struct.anon.0 }
%struct.anon.0 = type { %union.anon.1 }
%union.anon.1 = type { [2 x i8] }
%struct.small = type { i8 }

@.str = private unnamed_addr constant [4 x i8] c"%s\0A\00", align 1

; test1a: array of [16 x i8] 
;         no ssp attribute
; Requires no protector.
define void @test1a(i8* %a) nounwind uwtable {
entry:
; LINUX-I386: test1a:
; LINUX-I386-NOT: calll __stack_chk_fail
; LINUX-I386: .cfi_endproc

; LINUX-X64: test1a:
; LINUX-X64-NOT: callq __stack_chk_fail
; LINUX-X64: .cfi_endproc

; LINUX-KERNEL-X64: test1a:
; LINUX-KERNEL-X64-NOT: callq __stack_chk_fail
; LINUX-KERNEL-X64: .cfi_endproc

; DARWIN-X64: test1a:
; DARWIN-X64-NOT: callq ___stack_chk_fail
; DARWIN-X64: .cfi_endproc
  %a.addr = alloca i8*, align 8
  %buf = alloca [16 x i8], align 16
  store i8* %a, i8** %a.addr, align 8
  %arraydecay = getelementptr inbounds [16 x i8]* %buf, i32 0, i32 0
  %0 = load i8** %a.addr, align 8
  %call = call i8* @strcpy(i8* %arraydecay, i8* %0)
  %arraydecay1 = getelementptr inbounds [16 x i8]* %buf, i32 0, i32 0
  %call2 = call i32 (i8*, ...)* @printf(i8* getelementptr inbounds ([4 x i8]* @.str, i32 0, i32 0), i8* %arraydecay1)
  ret void
}

; test1b: array of [16 x i8] 
;         ssp attribute
; Requires protector.
define void @test1b(i8* %a) nounwind uwtable ssp {
entry:
; LINUX-I386: test1b:
; LINUX-I386: mov{{l|q}} %gs:
; LINUX-I386: calll __stack_chk_fail

; LINUX-X64: test1b:
; LINUX-X64: mov{{l|q}} %fs:
; LINUX-X64: callq __stack_chk_fail

; LINUX-KERNEL-X64: test1b:
; LINUX-KERNEL-X64: mov{{l|q}} %gs:
; LINUX-KERNEL-X64: callq __stack_chk_fail

; DARWIN-X64: test1b:
; DARWIN-X64: mov{{l|q}} ___stack_chk_guard
; DARWIN-X64: callq ___stack_chk_fail
  %a.addr = alloca i8*, align 8
  %buf = alloca [16 x i8], align 16
  store i8* %a, i8** %a.addr, align 8
  %arraydecay = getelementptr inbounds [16 x i8]* %buf, i32 0, i32 0
  %0 = load i8** %a.addr, align 8
  %call = call i8* @strcpy(i8* %arraydecay, i8* %0)
  %arraydecay1 = getelementptr inbounds [16 x i8]* %buf, i32 0, i32 0
  %call2 = call i32 (i8*, ...)* @printf(i8* getelementptr inbounds ([4 x i8]* @.str, i32 0, i32 0), i8* %arraydecay1)
  ret void
}

; test1c: array of [16 x i8] 
;         sspstrong attribute
; Requires protector.
define void @test1c(i8* %a) nounwind uwtable sspstrong {
entry:
; LINUX-I386: test1c:
; LINUX-I386: mov{{l|q}} %gs:
; LINUX-I386: calll __stack_chk_fail

; LINUX-X64: test1c:
; LINUX-X64: mov{{l|q}} %fs:
; LINUX-X64: callq __stack_chk_fail

; LINUX-KERNEL-X64: test1c:
; LINUX-KERNEL-X64: mov{{l|q}} %gs:
; LINUX-KERNEL-X64: callq __stack_chk_fail

; DARWIN-X64: test1c:
; DARWIN-X64: mov{{l|q}} ___stack_chk_guard
; DARWIN-X64: callq ___stack_chk_fail
  %a.addr = alloca i8*, align 8
  %buf = alloca [16 x i8], align 16
  store i8* %a, i8** %a.addr, align 8
  %arraydecay = getelementptr inbounds [16 x i8]* %buf, i32 0, i32 0
  %0 = load i8** %a.addr, align 8
  %call = call i8* @strcpy(i8* %arraydecay, i8* %0)
  %arraydecay1 = getelementptr inbounds [16 x i8]* %buf, i32 0, i32 0
  %call2 = call i32 (i8*, ...)* @printf(i8* getelementptr inbounds ([4 x i8]* @.str, i32 0, i32 0), i8* %arraydecay1)
  ret void
}

; test1d: array of [16 x i8] 
;         sspreq attribute
; Requires protector.
define void @test1d(i8* %a) nounwind uwtable sspreq {
entry:
; LINUX-I386: test1d:
; LINUX-I386: mov{{l|q}} %gs:
; LINUX-I386: calll __stack_chk_fail

; LINUX-X64: test1d:
; LINUX-X64: mov{{l|q}} %fs:
; LINUX-X64: callq __stack_chk_fail

; LINUX-KERNEL-X64: test1d:
; LINUX-KERNEL-X64: mov{{l|q}} %gs:
; LINUX-KERNEL-X64: callq __stack_chk_fail

; DARWIN-X64: test1d:
; DARWIN-X64: mov{{l|q}} ___stack_chk_guard
; DARWIN-X64: callq ___stack_chk_fail
  %a.addr = alloca i8*, align 8
  %buf = alloca [16 x i8], align 16
  store i8* %a, i8** %a.addr, align 8
  %arraydecay = getelementptr inbounds [16 x i8]* %buf, i32 0, i32 0
  %0 = load i8** %a.addr, align 8
  %call = call i8* @strcpy(i8* %arraydecay, i8* %0)
  %arraydecay1 = getelementptr inbounds [16 x i8]* %buf, i32 0, i32 0
  %call2 = call i32 (i8*, ...)* @printf(i8* getelementptr inbounds ([4 x i8]* @.str, i32 0, i32 0), i8* %arraydecay1)
  ret void
}

; test2a: struct { [16 x i8] }
;         no ssp attribute
; Requires no protector.
define void @test2a(i8* %a) nounwind uwtable {
entry:
; LINUX-I386: test2a:
; LINUX-I386-NOT: calll __stack_chk_fail
; LINUX-I386: .cfi_endproc

; LINUX-X64: test2a:
; LINUX-X64-NOT: callq __stack_chk_fail
; LINUX-X64: .cfi_endproc

; LINUX-KERNEL-X64: test2a:
; LINUX-KERNEL-X64-NOT: callq __stack_chk_fail
; LINUX-KERNEL-X64: .cfi_endproc

; DARWIN-X64: test2a:
; DARWIN-X64-NOT: callq ___stack_chk_fail
; DARWIN-X64: .cfi_endproc
  %a.addr = alloca i8*, align 8
  %b = alloca %struct.foo, align 1
  store i8* %a, i8** %a.addr, align 8
  %buf = getelementptr inbounds %struct.foo* %b, i32 0, i32 0
  %arraydecay = getelementptr inbounds [16 x i8]* %buf, i32 0, i32 0
  %0 = load i8** %a.addr, align 8
  %call = call i8* @strcpy(i8* %arraydecay, i8* %0)
  %buf1 = getelementptr inbounds %struct.foo* %b, i32 0, i32 0
  %arraydecay2 = getelementptr inbounds [16 x i8]* %buf1, i32 0, i32 0
  %call3 = call i32 (i8*, ...)* @printf(i8* getelementptr inbounds ([4 x i8]* @.str, i32 0, i32 0), i8* %arraydecay2)
  ret void
}

; test2b: struct { [16 x i8] }
;          ssp attribute
; Requires protector.
define void @test2b(i8* %a) nounwind uwtable ssp {
entry:
; LINUX-I386: test2b:
; LINUX-I386: mov{{l|q}} %gs:
; LINUX-I386: calll __stack_chk_fail

; LINUX-X64: test2b:
; LINUX-X64: mov{{l|q}} %fs:
; LINUX-X64: callq __stack_chk_fail

; LINUX-KERNEL-X64: test2b:
; LINUX-KERNEL-X64: mov{{l|q}} %gs:
; LINUX-KERNEL-X64: callq __stack_chk_fail

; DARWIN-X64: test2b:
; DARWIN-X64: mov{{l|q}} ___stack_chk_guard
; DARWIN-X64: callq ___stack_chk_fail
  %a.addr = alloca i8*, align 8
  %b = alloca %struct.foo, align 1
  store i8* %a, i8** %a.addr, align 8
  %buf = getelementptr inbounds %struct.foo* %b, i32 0, i32 0
  %arraydecay = getelementptr inbounds [16 x i8]* %buf, i32 0, i32 0
  %0 = load i8** %a.addr, align 8
  %call = call i8* @strcpy(i8* %arraydecay, i8* %0)
  %buf1 = getelementptr inbounds %struct.foo* %b, i32 0, i32 0
  %arraydecay2 = getelementptr inbounds [16 x i8]* %buf1, i32 0, i32 0
  %call3 = call i32 (i8*, ...)* @printf(i8* getelementptr inbounds ([4 x i8]* @.str, i32 0, i32 0), i8* %arraydecay2)
  ret void
}

; test2c: struct { [16 x i8] }
;          sspstrong attribute
; Requires protector.
define void @test2c(i8* %a) nounwind uwtable sspstrong {
entry:
; LINUX-I386: test2c:
; LINUX-I386: mov{{l|q}} %gs:
; LINUX-I386: calll __stack_chk_fail

; LINUX-X64: test2c:
; LINUX-X64: mov{{l|q}} %fs:
; LINUX-X64: callq __stack_chk_fail

; LINUX-KERNEL-X64: test2c:
; LINUX-KERNEL-X64: mov{{l|q}} %gs:
; LINUX-KERNEL-X64: callq __stack_chk_fail

; DARWIN-X64: test2c:
; DARWIN-X64: mov{{l|q}} ___stack_chk_guard
; DARWIN-X64: callq ___stack_chk_fail
  %a.addr = alloca i8*, align 8
  %b = alloca %struct.foo, align 1
  store i8* %a, i8** %a.addr, align 8
  %buf = getelementptr inbounds %struct.foo* %b, i32 0, i32 0
  %arraydecay = getelementptr inbounds [16 x i8]* %buf, i32 0, i32 0
  %0 = load i8** %a.addr, align 8
  %call = call i8* @strcpy(i8* %arraydecay, i8* %0)
  %buf1 = getelementptr inbounds %struct.foo* %b, i32 0, i32 0
  %arraydecay2 = getelementptr inbounds [16 x i8]* %buf1, i32 0, i32 0
  %call3 = call i32 (i8*, ...)* @printf(i8* getelementptr inbounds ([4 x i8]* @.str, i32 0, i32 0), i8* %arraydecay2)
  ret void
}

; test2d: struct { [16 x i8] }
;          sspreq attribute
; Requires protector.
define void @test2d(i8* %a) nounwind uwtable sspreq {
entry:
; LINUX-I386: test2d:
; LINUX-I386: mov{{l|q}} %gs:
; LINUX-I386: calll __stack_chk_fail

; LINUX-X64: test2d:
; LINUX-X64: mov{{l|q}} %fs:
; LINUX-X64: callq __stack_chk_fail

; LINUX-KERNEL-X64: test2d:
; LINUX-KERNEL-X64: mov{{l|q}} %gs:
; LINUX-KERNEL-X64: callq __stack_chk_fail

; DARWIN-X64: test2d:
; DARWIN-X64: mov{{l|q}} ___stack_chk_guard
; DARWIN-X64: callq ___stack_chk_fail
  %a.addr = alloca i8*, align 8
  %b = alloca %struct.foo, align 1
  store i8* %a, i8** %a.addr, align 8
  %buf = getelementptr inbounds %struct.foo* %b, i32 0, i32 0
  %arraydecay = getelementptr inbounds [16 x i8]* %buf, i32 0, i32 0
  %0 = load i8** %a.addr, align 8
  %call = call i8* @strcpy(i8* %arraydecay, i8* %0)
  %buf1 = getelementptr inbounds %struct.foo* %b, i32 0, i32 0
  %arraydecay2 = getelementptr inbounds [16 x i8]* %buf1, i32 0, i32 0
  %call3 = call i32 (i8*, ...)* @printf(i8* getelementptr inbounds ([4 x i8]* @.str, i32 0, i32 0), i8* %arraydecay2)
  ret void
}

; test3a:  array of [4 x i8]
;          no ssp attribute
; Requires no protector.
define void @test3a(i8* %a) nounwind uwtable {
entry:
; LINUX-I386: test3a:
; LINUX-I386-NOT: calll __stack_chk_fail
; LINUX-I386: .cfi_endproc

; LINUX-X64: test3a:
; LINUX-X64-NOT: callq __stack_chk_fail
; LINUX-X64: .cfi_endproc

; LINUX-KERNEL-X64: test3a:
; LINUX-KERNEL-X64-NOT: callq __stack_chk_fail
; LINUX-KERNEL-X64: .cfi_endproc

; DARWIN-X64: test3a:
; DARWIN-X64-NOT: callq ___stack_chk_fail
; DARWIN-X64: .cfi_endproc
  %a.addr = alloca i8*, align 8
  %buf = alloca [4 x i8], align 1
  store i8* %a, i8** %a.addr, align 8
  %arraydecay = getelementptr inbounds [4 x i8]* %buf, i32 0, i32 0
  %0 = load i8** %a.addr, align 8
  %call = call i8* @strcpy(i8* %arraydecay, i8* %0)
  %arraydecay1 = getelementptr inbounds [4 x i8]* %buf, i32 0, i32 0
  %call2 = call i32 (i8*, ...)* @printf(i8* getelementptr inbounds ([4 x i8]* @.str, i32 0, i32 0), i8* %arraydecay1)
  ret void
}

; test3b:  array [4 x i8]
;          ssp attribute
; Requires no protector.
define void @test3b(i8* %a) nounwind uwtable ssp {
entry:
; LINUX-I386: test3b:
; LINUX-I386-NOT: calll __stack_chk_fail
; LINUX-I386: .cfi_endproc

; LINUX-X64: test3b:
; LINUX-X64-NOT: callq __stack_chk_fail
; LINUX-X64: .cfi_endproc

; LINUX-KERNEL-X64: test3b:
; LINUX-KERNEL-X64-NOT: callq __stack_chk_fail
; LINUX-KERNEL-X64: .cfi_endproc

; DARWIN-X64: test3b:
; DARWIN-X64-NOT: callq ___stack_chk_fail
; DARWIN-X64: .cfi_endproc
  %a.addr = alloca i8*, align 8
  %buf = alloca [4 x i8], align 1
  store i8* %a, i8** %a.addr, align 8
  %arraydecay = getelementptr inbounds [4 x i8]* %buf, i32 0, i32 0
  %0 = load i8** %a.addr, align 8
  %call = call i8* @strcpy(i8* %arraydecay, i8* %0)
  %arraydecay1 = getelementptr inbounds [4 x i8]* %buf, i32 0, i32 0
  %call2 = call i32 (i8*, ...)* @printf(i8* getelementptr inbounds ([4 x i8]* @.str, i32 0, i32 0), i8* %arraydecay1)
  ret void
}

; test3c:  array of [4 x i8]
;          sspstrong attribute
; Requires protector.
define void @test3c(i8* %a) nounwind uwtable sspstrong {
entry:
; LINUX-I386: test3c:
; LINUX-I386: mov{{l|q}} %gs:
; LINUX-I386: calll __stack_chk_fail

; LINUX-X64: test3c:
; LINUX-X64: mov{{l|q}} %fs:
; LINUX-X64: callq __stack_chk_fail

; LINUX-KERNEL-X64: test3c:
; LINUX-KERNEL-X64: mov{{l|q}} %gs:
; LINUX-KERNEL-X64: callq __stack_chk_fail

; DARWIN-X64: test3c:
; DARWIN-X64: mov{{l|q}} ___stack_chk_guard
; DARWIN-X64: callq ___stack_chk_fail
  %a.addr = alloca i8*, align 8
  %buf = alloca [4 x i8], align 1
  store i8* %a, i8** %a.addr, align 8
  %arraydecay = getelementptr inbounds [4 x i8]* %buf, i32 0, i32 0
  %0 = load i8** %a.addr, align 8
  %call = call i8* @strcpy(i8* %arraydecay, i8* %0)
  %arraydecay1 = getelementptr inbounds [4 x i8]* %buf, i32 0, i32 0
  %call2 = call i32 (i8*, ...)* @printf(i8* getelementptr inbounds ([4 x i8]* @.str, i32 0, i32 0), i8* %arraydecay1)
  ret void
}

; test3d:  array of [4 x i8]
;          sspreq attribute
; Requires protector.
define void @test3d(i8* %a) nounwind uwtable sspreq {
entry:
; LINUX-I386: test3d:
; LINUX-I386: mov{{l|q}} %gs:
; LINUX-I386: calll __stack_chk_fail

; LINUX-X64: test3d:
; LINUX-X64: mov{{l|q}} %fs:
; LINUX-X64: callq __stack_chk_fail

; LINUX-KERNEL-X64: test3d:
; LINUX-KERNEL-X64: mov{{l|q}} %gs:
; LINUX-KERNEL-X64: callq __stack_chk_fail

; DARWIN-X64: test3d:
; DARWIN-X64: mov{{l|q}} ___stack_chk_guard
; DARWIN-X64: callq ___stack_chk_fail
  %a.addr = alloca i8*, align 8
  %buf = alloca [4 x i8], align 1
  store i8* %a, i8** %a.addr, align 8
  %arraydecay = getelementptr inbounds [4 x i8]* %buf, i32 0, i32 0
  %0 = load i8** %a.addr, align 8
  %call = call i8* @strcpy(i8* %arraydecay, i8* %0)
  %arraydecay1 = getelementptr inbounds [4 x i8]* %buf, i32 0, i32 0
  %call2 = call i32 (i8*, ...)* @printf(i8* getelementptr inbounds ([4 x i8]* @.str, i32 0, i32 0), i8* %arraydecay1)
  ret void
}

; test4a:  struct { [4 x i8] }
;          no ssp attribute
; Requires no protector.
define void @test4a(i8* %a) nounwind uwtable {
entry:
; LINUX-I386: test4a:
; LINUX-I386-NOT: calll __stack_chk_fail
; LINUX-I386: .cfi_endproc

; LINUX-X64: test4a:
; LINUX-X64-NOT: callq __stack_chk_fail
; LINUX-X64: .cfi_endproc

; LINUX-KERNEL-X64: test4a:
; LINUX-KERNEL-X64-NOT: callq __stack_chk_fail
; LINUX-KERNEL-X64: .cfi_endproc

; DARWIN-X64: test4a:
; DARWIN-X64-NOT: callq ___stack_chk_fail
; DARWIN-X64: .cfi_endproc
  %a.addr = alloca i8*, align 8
  %b = alloca %struct.foo.0, align 1
  store i8* %a, i8** %a.addr, align 8
  %buf = getelementptr inbounds %struct.foo.0* %b, i32 0, i32 0
  %arraydecay = getelementptr inbounds [4 x i8]* %buf, i32 0, i32 0
  %0 = load i8** %a.addr, align 8
  %call = call i8* @strcpy(i8* %arraydecay, i8* %0)
  %buf1 = getelementptr inbounds %struct.foo.0* %b, i32 0, i32 0
  %arraydecay2 = getelementptr inbounds [4 x i8]* %buf1, i32 0, i32 0
  %call3 = call i32 (i8*, ...)* @printf(i8* getelementptr inbounds ([4 x i8]* @.str, i32 0, i32 0), i8* %arraydecay2)
  ret void
}

; test4b:  struct { [4 x i8] }
;          ssp attribute
; Requires no protector.
define void @test4b(i8* %a) nounwind uwtable ssp {
entry:
; LINUX-I386: test4b:
; LINUX-I386-NOT: calll __stack_chk_fail
; LINUX-I386: .cfi_endproc

; LINUX-X64: test4b:
; LINUX-X64-NOT: callq __stack_chk_fail
; LINUX-X64: .cfi_endproc

; LINUX-KERNEL-X64: test4b:
; LINUX-KERNEL-X64-NOT: callq __stack_chk_fail
; LINUX-KERNEL-X64: .cfi_endproc

; DARWIN-X64: test4b:
; DARWIN-X64-NOT: callq ___stack_chk_fail
; DARWIN-X64: .cfi_endproc
  %a.addr = alloca i8*, align 8
  %b = alloca %struct.foo.0, align 1
  store i8* %a, i8** %a.addr, align 8
  %buf = getelementptr inbounds %struct.foo.0* %b, i32 0, i32 0
  %arraydecay = getelementptr inbounds [4 x i8]* %buf, i32 0, i32 0
  %0 = load i8** %a.addr, align 8
  %call = call i8* @strcpy(i8* %arraydecay, i8* %0)
  %buf1 = getelementptr inbounds %struct.foo.0* %b, i32 0, i32 0
  %arraydecay2 = getelementptr inbounds [4 x i8]* %buf1, i32 0, i32 0
  %call3 = call i32 (i8*, ...)* @printf(i8* getelementptr inbounds ([4 x i8]* @.str, i32 0, i32 0), i8* %arraydecay2)
  ret void
}

; test4c:  struct { [4 x i8] }
;          sspstrong attribute
; Requires protector.
define void @test4c(i8* %a) nounwind uwtable sspstrong {
entry:
; LINUX-I386: test4c:
; LINUX-I386: mov{{l|q}} %gs:
; LINUX-I386: calll __stack_chk_fail

; LINUX-X64: test4c:
; LINUX-X64: mov{{l|q}} %fs:
; LINUX-X64: callq __stack_chk_fail

; LINUX-KERNEL-X64: test4c:
; LINUX-KERNEL-X64: mov{{l|q}} %gs:
; LINUX-KERNEL-X64: callq __stack_chk_fail

; DARWIN-X64: test4c:
; DARWIN-X64: mov{{l|q}} ___stack_chk_guard
; DARWIN-X64: callq ___stack_chk_fail
  %a.addr = alloca i8*, align 8
  %b = alloca %struct.foo.0, align 1
  store i8* %a, i8** %a.addr, align 8
  %buf = getelementptr inbounds %struct.foo.0* %b, i32 0, i32 0
  %arraydecay = getelementptr inbounds [4 x i8]* %buf, i32 0, i32 0
  %0 = load i8** %a.addr, align 8
  %call = call i8* @strcpy(i8* %arraydecay, i8* %0)
  %buf1 = getelementptr inbounds %struct.foo.0* %b, i32 0, i32 0
  %arraydecay2 = getelementptr inbounds [4 x i8]* %buf1, i32 0, i32 0
  %call3 = call i32 (i8*, ...)* @printf(i8* getelementptr inbounds ([4 x i8]* @.str, i32 0, i32 0), i8* %arraydecay2)
  ret void
}

; test4d:  struct { [4 x i8] }
;          sspreq attribute
; Requires protector.
define void @test4d(i8* %a) nounwind uwtable sspreq {
entry:
; LINUX-I386: test4d:
; LINUX-I386: mov{{l|q}} %gs:
; LINUX-I386: calll __stack_chk_fail

; LINUX-X64: test4d:
; LINUX-X64: mov{{l|q}} %fs:
; LINUX-X64: callq __stack_chk_fail

; LINUX-KERNEL-X64: test4d:
; LINUX-KERNEL-X64: mov{{l|q}} %gs:
; LINUX-KERNEL-X64: callq __stack_chk_fail

; DARWIN-X64: test4d:
; DARWIN-X64: mov{{l|q}} ___stack_chk_guard
; DARWIN-X64: callq ___stack_chk_fail
  %a.addr = alloca i8*, align 8
  %b = alloca %struct.foo.0, align 1
  store i8* %a, i8** %a.addr, align 8
  %buf = getelementptr inbounds %struct.foo.0* %b, i32 0, i32 0
  %arraydecay = getelementptr inbounds [4 x i8]* %buf, i32 0, i32 0
  %0 = load i8** %a.addr, align 8
  %call = call i8* @strcpy(i8* %arraydecay, i8* %0)
  %buf1 = getelementptr inbounds %struct.foo.0* %b, i32 0, i32 0
  %arraydecay2 = getelementptr inbounds [4 x i8]* %buf1, i32 0, i32 0
  %call3 = call i32 (i8*, ...)* @printf(i8* getelementptr inbounds ([4 x i8]* @.str, i32 0, i32 0), i8* %arraydecay2)
  ret void
}

; test5a:  no arrays / no nested arrays
;          no ssp attribute
; Requires no protector.
define void @test5a(i8* %a) nounwind uwtable {
entry:
; LINUX-I386: test5a:
; LINUX-I386-NOT: calll __stack_chk_fail
; LINUX-I386: .cfi_endproc

; LINUX-X64: test5a:
; LINUX-X64-NOT: callq __stack_chk_fail
; LINUX-X64: .cfi_endproc

; LINUX-KERNEL-X64: test5a:
; LINUX-KERNEL-X64-NOT: callq __stack_chk_fail
; LINUX-KERNEL-X64: .cfi_endproc

; DARWIN-X64: test5a:
; DARWIN-X64-NOT: callq ___stack_chk_fail
; DARWIN-X64: .cfi_endproc
  %a.addr = alloca i8*, align 8
  store i8* %a, i8** %a.addr, align 8
  %0 = load i8** %a.addr, align 8
  %call = call i32 (i8*, ...)* @printf(i8* getelementptr inbounds ([4 x i8]* @.str, i32 0, i32 0), i8* %0)
  ret void
}

; test5b:  no arrays / no nested arrays
;          ssp attribute
; Requires no protector.
define void @test5b(i8* %a) nounwind uwtable ssp {
entry:
; LINUX-I386: test5b:
; LINUX-I386-NOT: calll __stack_chk_fail
; LINUX-I386: .cfi_endproc

; LINUX-X64: test5b:
; LINUX-X64-NOT: callq __stack_chk_fail
; LINUX-X64: .cfi_endproc

; LINUX-KERNEL-X64: test5b:
; LINUX-KERNEL-X64-NOT: callq __stack_chk_fail
; LINUX-KERNEL-X64: .cfi_endproc

; DARWIN-X64: test5b:
; DARWIN-X64-NOT: callq ___stack_chk_fail
; DARWIN-X64: .cfi_endproc
  %a.addr = alloca i8*, align 8
  store i8* %a, i8** %a.addr, align 8
  %0 = load i8** %a.addr, align 8
  %call = call i32 (i8*, ...)* @printf(i8* getelementptr inbounds ([4 x i8]* @.str, i32 0, i32 0), i8* %0)
  ret void
}

; test5c:  no arrays / no nested arrays
;          sspstrong attribute
; Requires no protector.
define void @test5c(i8* %a) nounwind uwtable sspstrong {
entry:
; LINUX-I386: test5c:
; LINUX-I386-NOT: calll __stack_chk_fail
; LINUX-I386: .cfi_endproc

; LINUX-X64: test5c:
; LINUX-X64-NOT: callq __stack_chk_fail
; LINUX-X64: .cfi_endproc

; LINUX-KERNEL-X64: test5c:
; LINUX-KERNEL-X64-NOT: callq __stack_chk_fail
; LINUX-KERNEL-X64: .cfi_endproc

; DARWIN-X64: test5c:
; DARWIN-X64-NOT: callq ___stack_chk_fail
; DARWIN-X64: .cfi_endproc
  %a.addr = alloca i8*, align 8
  store i8* %a, i8** %a.addr, align 8
  %0 = load i8** %a.addr, align 8
  %call = call i32 (i8*, ...)* @printf(i8* getelementptr inbounds ([4 x i8]* @.str, i32 0, i32 0), i8* %0)
  ret void
}

; test5d:  no arrays / no nested arrays
;          sspreq attribute
; Requires protector.
define void @test5d(i8* %a) nounwind uwtable sspreq {
entry:
; LINUX-I386: test5d:
; LINUX-I386: mov{{l|q}} %gs:
; LINUX-I386: calll __stack_chk_fail

; LINUX-X64: test5d:
; LINUX-X64: mov{{l|q}} %fs:
; LINUX-X64: callq __stack_chk_fail

; LINUX-KERNEL-X64: test5d:
; LINUX-KERNEL-X64: mov{{l|q}} %gs:
; LINUX-KERNEL-X64: callq __stack_chk_fail

; DARWIN-X64: test5d:
; DARWIN-X64: mov{{l|q}} ___stack_chk_guard
; DARWIN-X64: callq ___stack_chk_fail
  %a.addr = alloca i8*, align 8
  store i8* %a, i8** %a.addr, align 8
  %0 = load i8** %a.addr, align 8
  %call = call i32 (i8*, ...)* @printf(i8* getelementptr inbounds ([4 x i8]* @.str, i32 0, i32 0), i8* %0)
  ret void
}

; test6a:  Address-of local taken (j = &a)
;          no ssp attribute
; Requires no protector.
define void @test6a() nounwind uwtable {
entry:
; LINUX-I386: test6a:
; LINUX-I386-NOT: calll __stack_chk_fail
; LINUX-I386: .cfi_endproc

; LINUX-X64: test6a:
; LINUX-X64-NOT: callq __stack_chk_fail
; LINUX-X64: .cfi_endproc

; LINUX-KERNEL-X64: test6a:
; LINUX-KERNEL-X64-NOT: callq __stack_chk_fail
; LINUX-KERNEL-X64: .cfi_endproc

; DARWIN-X64: test6a:
; DARWIN-X64-NOT: callq ___stack_chk_fail
; DARWIN-X64: .cfi_endproc
  %retval = alloca i32, align 4
  %a = alloca i32, align 4
  %j = alloca i32*, align 8
  store i32 0, i32* %retval
  %0 = load i32* %a, align 4
  %add = add nsw i32 %0, 1
  store i32 %add, i32* %a, align 4
  store i32* %a, i32** %j, align 8
  ret void
}

; test6b:  Address-of local taken (j = &a)
;          ssp attribute
; Requires no protector.
define void @test6b() nounwind uwtable ssp {
entry:
; LINUX-I386: test6b:
; LINUX-I386-NOT: calll __stack_chk_fail
; LINUX-I386: .cfi_endproc

; LINUX-X64: test6b:
; LINUX-X64-NOT: callq __stack_chk_fail
; LINUX-X64: .cfi_endproc

; LINUX-KERNEL-X64: test6b:
; LINUX-KERNEL-X64-NOT: callq __stack_chk_fail
; LINUX-KERNEL-X64: .cfi_endproc

; DARWIN-X64: test6b:
; DARWIN-X64-NOT: callq ___stack_chk_fail
; DARWIN-X64: .cfi_endproc
  %retval = alloca i32, align 4
  %a = alloca i32, align 4
  %j = alloca i32*, align 8
  store i32 0, i32* %retval
  %0 = load i32* %a, align 4
  %add = add nsw i32 %0, 1
  store i32 %add, i32* %a, align 4
  store i32* %a, i32** %j, align 8
  ret void
}

; test6c:  Address-of local taken (j = &a)
;          sspstrong attribute
; Requires protector.
define void @test6c() nounwind uwtable sspstrong {
entry:
; LINUX-I386: test6c:
; LINUX-I386: mov{{l|q}} %gs:
; LINUX-I386: calll __stack_chk_fail

; LINUX-X64: test6c:
; LINUX-X64: mov{{l|q}} %fs:
; LINUX-X64: callq __stack_chk_fail

; LINUX-KERNEL-X64: test6c:
; LINUX-KERNEL-X64: mov{{l|q}} %gs:
; LINUX-KERNEL-X64: callq __stack_chk_fail

; DARWIN-X64: test6c:
; DARWIN-X64: mov{{l|q}} ___stack_chk_guard
; DARWIN-X64: callq ___stack_chk_fail
  %retval = alloca i32, align 4
  %a = alloca i32, align 4
  %j = alloca i32*, align 8
  store i32 0, i32* %retval
  %0 = load i32* %a, align 4
  %add = add nsw i32 %0, 1
  store i32 %add, i32* %a, align 4
  store i32* %a, i32** %j, align 8
  ret void
}

; test6d:  Address-of local taken (j = &a)
;          sspreq attribute
; Requires protector.
define void @test6d() nounwind uwtable sspreq {
entry:
; LINUX-I386: test6d:
; LINUX-I386: mov{{l|q}} %gs:
; LINUX-I386: calll __stack_chk_fail

; LINUX-X64: test6d:
; LINUX-X64: mov{{l|q}} %fs:
; LINUX-X64: callq __stack_chk_fail

; LINUX-KERNEL-X64: test6d:
; LINUX-KERNEL-X64: mov{{l|q}} %gs:
; LINUX-KERNEL-X64: callq __stack_chk_fail

; DARWIN-X64: test6d:
; DARWIN-X64: mov{{l|q}} ___stack_chk_guard
; DARWIN-X64: callq ___stack_chk_fail
  %retval = alloca i32, align 4
  %a = alloca i32, align 4
  %j = alloca i32*, align 8
  store i32 0, i32* %retval
  %0 = load i32* %a, align 4
  %add = add nsw i32 %0, 1
  store i32 %add, i32* %a, align 4
  store i32* %a, i32** %j, align 8
  ret void
}

; test7a:  PtrToInt Cast
;          no ssp attribute
; Requires no protector.
define void @test7a() nounwind uwtable readnone {
entry:
; LINUX-I386: test7a:
; LINUX-I386-NOT: calll __stack_chk_fail
; LINUX-I386: .cfi_endproc

; LINUX-X64: test7a:
; LINUX-X64-NOT: callq __stack_chk_fail
; LINUX-X64: .cfi_endproc

; LINUX-KERNEL-X64: test7a:
; LINUX-KERNEL-X64-NOT: callq __stack_chk_fail
; LINUX-KERNEL-X64: .cfi_endproc

; DARWIN-X64: test7a:
; DARWIN-X64-NOT: callq ___stack_chk_fail
; DARWIN-X64: .cfi_endproc
  %a = alloca i32, align 4
  %0 = ptrtoint i32* %a to i64
  %call = call i32 (i8*, ...)* @printf(i8* getelementptr inbounds ([4 x i8]* @.str, i32 0, i32 0), i64 %0)
  ret void
}

; test7b:  PtrToInt Cast
;          ssp attribute
; Requires no protector.
define void @test7b() nounwind uwtable readnone ssp {
entry:
; LINUX-I386: test7b:
; LINUX-I386-NOT: calll __stack_chk_fail
; LINUX-I386: .cfi_endproc

; LINUX-X64: test7b:
; LINUX-X64-NOT: callq __stack_chk_fail
; LINUX-X64: .cfi_endproc

; LINUX-KERNEL-X64: test7b:
; LINUX-KERNEL-X64-NOT: callq __stack_chk_fail
; LINUX-KERNEL-X64: .cfi_endproc

; DARWIN-X64: test7b:
; DARWIN-X64-NOT: callq ___stack_chk_fail
; DARWIN-X64: .cfi_endproc
  %a = alloca i32, align 4
  %0 = ptrtoint i32* %a to i64
  %call = call i32 (i8*, ...)* @printf(i8* getelementptr inbounds ([4 x i8]* @.str, i32 0, i32 0), i64 %0)
  ret void
}

; test7c:  PtrToInt Cast
;          sspstrong attribute
; Requires protector.
define void @test7c() nounwind uwtable readnone sspstrong {
entry:
; LINUX-I386: test7c:
; LINUX-I386: mov{{l|q}} %gs:
; LINUX-I386: calll __stack_chk_fail

; LINUX-X64: test7c:
; LINUX-X64: mov{{l|q}} %fs:
; LINUX-X64: callq __stack_chk_fail

; LINUX-KERNEL-X64: test7c:
; LINUX-KERNEL-X64: mov{{l|q}} %gs:
; LINUX-KERNEL-X64: callq __stack_chk_fail

; DARWIN-X64: test7c:
; DARWIN-X64: mov{{l|q}} ___stack_chk_guard
; DARWIN-X64: callq ___stack_chk_fail
  %a = alloca i32, align 4
  %0 = ptrtoint i32* %a to i64
  %call = call i32 (i8*, ...)* @printf(i8* getelementptr inbounds ([4 x i8]* @.str, i32 0, i32 0), i64 %0)
  ret void
}

; test7d:  PtrToInt Cast
;          sspreq attribute
; Requires protector.
define void @test7d() nounwind uwtable readnone sspreq {
entry:
; LINUX-I386: test7d:
; LINUX-I386: mov{{l|q}} %gs:
; LINUX-I386: calll __stack_chk_fail

; LINUX-X64: test7d:
; LINUX-X64: mov{{l|q}} %fs:
; LINUX-X64: callq __stack_chk_fail

; LINUX-KERNEL-X64: test7d:
; LINUX-KERNEL-X64: mov{{l|q}} %gs:
; LINUX-KERNEL-X64: callq __stack_chk_fail

; DARWIN-X64: test7d:
; DARWIN-X64: mov{{l|q}} ___stack_chk_guard
; DARWIN-X64: callq ___stack_chk_fail
  %a = alloca i32, align 4
  %0 = ptrtoint i32* %a to i64
  %call = call i32 (i8*, ...)* @printf(i8* getelementptr inbounds ([4 x i8]* @.str, i32 0, i32 0), i64 %0)
  ret void
}

; test8a:  Passing addr-of to function call
;          no ssp attribute
; Requires no protector.
define void @test8a() nounwind uwtable {
entry:
; LINUX-I386: test8a:
; LINUX-I386-NOT: calll __stack_chk_fail
; LINUX-I386: .cfi_endproc

; LINUX-X64: test8a:
; LINUX-X64-NOT: callq __stack_chk_fail
; LINUX-X64: .cfi_endproc

; LINUX-KERNEL-X64: test8a:
; LINUX-KERNEL-X64-NOT: callq __stack_chk_fail
; LINUX-KERNEL-X64: .cfi_endproc

; DARWIN-X64: test8a:
; DARWIN-X64-NOT: callq ___stack_chk_fail
; DARWIN-X64: .cfi_endproc
  %b = alloca i32, align 4
  call void @funcall(i32* %b) nounwind
  ret void
}

; test8b:  Passing addr-of to function call
;          ssp attribute
; Requires no protector.
define void @test8b() nounwind uwtable ssp {
entry:
; LINUX-I386: test8b:
; LINUX-I386-NOT: calll __stack_chk_fail
; LINUX-I386: .cfi_endproc

; LINUX-X64: test8b:
; LINUX-X64-NOT: callq __stack_chk_fail
; LINUX-X64: .cfi_endproc

; LINUX-KERNEL-X64: test8b:
; LINUX-KERNEL-X64-NOT: callq __stack_chk_fail
; LINUX-KERNEL-X64: .cfi_endproc

; DARWIN-X64: test8b:
; DARWIN-X64-NOT: callq ___stack_chk_fail
; DARWIN-X64: .cfi_endproc
  %b = alloca i32, align 4
  call void @funcall(i32* %b) nounwind
  ret void
}

; test8c:  Passing addr-of to function call
;          sspstrong attribute
; Requires protector.
define void @test8c() nounwind uwtable sspstrong {
entry:
; LINUX-I386: test8c:
; LINUX-I386: mov{{l|q}} %gs:
; LINUX-I386: calll __stack_chk_fail

; LINUX-X64: test8c:
; LINUX-X64: mov{{l|q}} %fs:
; LINUX-X64: callq __stack_chk_fail

; LINUX-KERNEL-X64: test8c:
; LINUX-KERNEL-X64: mov{{l|q}} %gs:
; LINUX-KERNEL-X64: callq __stack_chk_fail

; DARWIN-X64: test8c:
; DARWIN-X64: mov{{l|q}} ___stack_chk_guard
; DARWIN-X64: callq ___stack_chk_fail
  %b = alloca i32, align 4
  call void @funcall(i32* %b) nounwind
  ret void
}

; test8d:  Passing addr-of to function call
;          sspreq attribute
; Requires protector.
define void @test8d() nounwind uwtable sspreq {
entry:
; LINUX-I386: test8d:
; LINUX-I386: mov{{l|q}} %gs:
; LINUX-I386: calll __stack_chk_fail

; LINUX-X64: test8d:
; LINUX-X64: mov{{l|q}} %fs:
; LINUX-X64: callq __stack_chk_fail

; LINUX-KERNEL-X64: test8d:
; LINUX-KERNEL-X64: mov{{l|q}} %gs:
; LINUX-KERNEL-X64: callq __stack_chk_fail

; DARWIN-X64: test8d:
; DARWIN-X64: mov{{l|q}} ___stack_chk_guard
; DARWIN-X64: callq ___stack_chk_fail
  %b = alloca i32, align 4
  call void @funcall(i32* %b) nounwind
  ret void
}

; test9a:  Addr-of in select instruction
;          no ssp attribute
; Requires no protector.
define void @test9a() nounwind uwtable {
entry:
; LINUX-I386: test9a:
; LINUX-I386-NOT: calll __stack_chk_fail
; LINUX-I386: .cfi_endproc

; LINUX-X64: test9a:
; LINUX-X64-NOT: callq __stack_chk_fail
; LINUX-X64: .cfi_endproc

; LINUX-KERNEL-X64: test9a:
; LINUX-KERNEL-X64-NOT: callq __stack_chk_fail
; LINUX-KERNEL-X64: .cfi_endproc

; DARWIN-X64: test9a:
; DARWIN-X64-NOT: callq ___stack_chk_fail
; DARWIN-X64: .cfi_endproc
  %x = alloca double, align 8
  %call = call double @testi_aux() nounwind
  store double %call, double* %x, align 8
  %cmp2 = fcmp ogt double %call, 0.000000e+00
  %y.1 = select i1 %cmp2, double* %x, double* null
  %call2 = call i32 (i8*, ...)* @printf(i8* getelementptr inbounds ([4 x i8]* @.str, i32 0, i32 0), double* %y.1)
  ret void
}

; test9b:  Addr-of in select instruction
;          ssp attribute
; Requires no protector.
define void @test9b() nounwind uwtable ssp {
entry:
; LINUX-I386: test9b:
; LINUX-I386-NOT: calll __stack_chk_fail
; LINUX-I386: .cfi_endproc

; LINUX-X64: test9b:
; LINUX-X64-NOT: callq __stack_chk_fail
; LINUX-X64: .cfi_endproc

; LINUX-KERNEL-X64: test9b:
; LINUX-KERNEL-X64-NOT: callq __stack_chk_fail
; LINUX-KERNEL-X64: .cfi_endproc

; DARWIN-X64: test9b:
; DARWIN-X64-NOT: callq ___stack_chk_fail
; DARWIN-X64: .cfi_endproc
  %x = alloca double, align 8
  %call = call double @testi_aux() nounwind
  store double %call, double* %x, align 8
  %cmp2 = fcmp ogt double %call, 0.000000e+00
  %y.1 = select i1 %cmp2, double* %x, double* null
  %call2 = call i32 (i8*, ...)* @printf(i8* getelementptr inbounds ([4 x i8]* @.str, i32 0, i32 0), double* %y.1)
  ret void
}

; test9c:  Addr-of in select instruction
;          sspstrong attribute
; Requires protector.
define void @test9c() nounwind uwtable sspstrong {
entry:
; LINUX-I386: test9c:
; LINUX-I386: mov{{l|q}} %gs:
; LINUX-I386: calll __stack_chk_fail

; LINUX-X64: test9c:
; LINUX-X64: mov{{l|q}} %fs:
; LINUX-X64: callq __stack_chk_fail

; LINUX-KERNEL-X64: test9c:
; LINUX-KERNEL-X64: mov{{l|q}} %gs:
; LINUX-KERNEL-X64: callq __stack_chk_fail

; DARWIN-X64: test9c:
; DARWIN-X64: mov{{l|q}} ___stack_chk_guard
; DARWIN-X64: callq ___stack_chk_fail
  %x = alloca double, align 8
  %call = call double @testi_aux() nounwind
  store double %call, double* %x, align 8
  %cmp2 = fcmp ogt double %call, 0.000000e+00
  %y.1 = select i1 %cmp2, double* %x, double* null
  %call2 = call i32 (i8*, ...)* @printf(i8* getelementptr inbounds ([4 x i8]* @.str, i32 0, i32 0), double* %y.1)
  ret void
}

; test9d:  Addr-of in select instruction
;          sspreq attribute
; Requires protector.
define void @test9d() nounwind uwtable sspreq {
entry:
; LINUX-I386: test9d:
; LINUX-I386: mov{{l|q}} %gs:
; LINUX-I386: calll __stack_chk_fail

; LINUX-X64: test9d:
; LINUX-X64: mov{{l|q}} %fs:
; LINUX-X64: callq __stack_chk_fail

; LINUX-KERNEL-X64: test9d:
; LINUX-KERNEL-X64: mov{{l|q}} %gs:
; LINUX-KERNEL-X64: callq __stack_chk_fail

; DARWIN-X64: test9d:
; DARWIN-X64: mov{{l|q}} ___stack_chk_guard
; DARWIN-X64: callq ___stack_chk_fail
  %x = alloca double, align 8
  %call = call double @testi_aux() nounwind
  store double %call, double* %x, align 8
  %cmp2 = fcmp ogt double %call, 0.000000e+00
  %y.1 = select i1 %cmp2, double* %x, double* null
  %call2 = call i32 (i8*, ...)* @printf(i8* getelementptr inbounds ([4 x i8]* @.str, i32 0, i32 0), double* %y.1)
  ret void
}

; test10a: Addr-of in phi instruction
;          no ssp attribute
; Requires no protector.
define void @test10a() nounwind uwtable {
entry:
; LINUX-I386: test10a:
; LINUX-I386-NOT: calll __stack_chk_fail
; LINUX-I386: .cfi_endproc

; LINUX-X64: test10a:
; LINUX-X64-NOT: callq __stack_chk_fail
; LINUX-X64: .cfi_endproc

; LINUX-KERNEL-X64: test10a:
; LINUX-KERNEL-X64-NOT: callq __stack_chk_fail
; LINUX-KERNEL-X64: .cfi_endproc

; DARWIN-X64: test10a:
; DARWIN-X64-NOT: callq ___stack_chk_fail
; DARWIN-X64: .cfi_endproc
  %x = alloca double, align 8
  %call = call double @testi_aux() nounwind
  store double %call, double* %x, align 8
  %cmp = fcmp ogt double %call, 3.140000e+00
  br i1 %cmp, label %if.then, label %if.else

if.then:                                          ; preds = %entry
  %call1 = call double @testi_aux() nounwind
  store double %call1, double* %x, align 8
  br label %if.end4

if.else:                                          ; preds = %entry
  %cmp2 = fcmp ogt double %call, 1.000000e+00
  br i1 %cmp2, label %if.then3, label %if.end4

if.then3:                                         ; preds = %if.else
  br label %if.end4

if.end4:                                          ; preds = %if.else, %if.then3, %if.then
  %y.0 = phi double* [ null, %if.then ], [ %x, %if.then3 ], [ null, %if.else ]
  %call5 = call i32 (i8*, ...)* @printf(i8* getelementptr inbounds ([4 x i8]* @.str, i64 0, i64 0), double* %y.0) nounwind
  ret void
}

; test10b: Addr-of in phi instruction
;          ssp attribute
; Requires no protector.
define void @test10b() nounwind uwtable ssp {
entry:
; LINUX-I386: test10b:
; LINUX-I386-NOT: calll __stack_chk_fail
; LINUX-I386: .cfi_endproc

; LINUX-X64: test10b:
; LINUX-X64-NOT: callq __stack_chk_fail
; LINUX-X64: .cfi_endproc

; LINUX-KERNEL-X64: test10b:
; LINUX-KERNEL-X64-NOT: callq __stack_chk_fail
; LINUX-KERNEL-X64: .cfi_endproc

; DARWIN-X64: test10b:
; DARWIN-X64-NOT: callq ___stack_chk_fail
; DARWIN-X64: .cfi_endproc
  %x = alloca double, align 8
  %call = call double @testi_aux() nounwind
  store double %call, double* %x, align 8
  %cmp = fcmp ogt double %call, 3.140000e+00
  br i1 %cmp, label %if.then, label %if.else

if.then:                                          ; preds = %entry
  %call1 = call double @testi_aux() nounwind
  store double %call1, double* %x, align 8
  br label %if.end4

if.else:                                          ; preds = %entry
  %cmp2 = fcmp ogt double %call, 1.000000e+00
  br i1 %cmp2, label %if.then3, label %if.end4

if.then3:                                         ; preds = %if.else
  br label %if.end4

if.end4:                                          ; preds = %if.else, %if.then3, %if.then
  %y.0 = phi double* [ null, %if.then ], [ %x, %if.then3 ], [ null, %if.else ]
  %call5 = call i32 (i8*, ...)* @printf(i8* getelementptr inbounds ([4 x i8]* @.str, i64 0, i64 0), double* %y.0) nounwind
  ret void
}

; test10c: Addr-of in phi instruction
;          sspstrong attribute
; Requires protector.
define void @test10c() nounwind uwtable sspstrong {
entry:
; LINUX-I386: test10c:
; LINUX-I386: mov{{l|q}} %gs:
; LINUX-I386: calll __stack_chk_fail

; LINUX-X64: test10c:
; LINUX-X64: mov{{l|q}} %fs:
; LINUX-X64: callq __stack_chk_fail

; LINUX-KERNEL-X64: test10c:
; LINUX-KERNEL-X64: mov{{l|q}} %gs:
; LINUX-KERNEL-X64: callq __stack_chk_fail

; DARWIN-X64: test10c:
; DARWIN-X64: mov{{l|q}} ___stack_chk_guard
; DARWIN-X64: callq ___stack_chk_fail
  %x = alloca double, align 8
  %call = call double @testi_aux() nounwind
  store double %call, double* %x, align 8
  %cmp = fcmp ogt double %call, 3.140000e+00
  br i1 %cmp, label %if.then, label %if.else

if.then:                                          ; preds = %entry
  %call1 = call double @testi_aux() nounwind
  store double %call1, double* %x, align 8
  br label %if.end4

if.else:                                          ; preds = %entry
  %cmp2 = fcmp ogt double %call, 1.000000e+00
  br i1 %cmp2, label %if.then3, label %if.end4

if.then3:                                         ; preds = %if.else
  br label %if.end4

if.end4:                                          ; preds = %if.else, %if.then3, %if.then
  %y.0 = phi double* [ null, %if.then ], [ %x, %if.then3 ], [ null, %if.else ]
  %call5 = call i32 (i8*, ...)* @printf(i8* getelementptr inbounds ([4 x i8]* @.str, i64 0, i64 0), double* %y.0) nounwind
  ret void
}

; test10d: Addr-of in phi instruction
;          sspreq attribute
; Requires protector.
define void @test10d() nounwind uwtable sspreq {
entry:
; LINUX-I386: test10d:
; LINUX-I386: mov{{l|q}} %gs:
; LINUX-I386: calll __stack_chk_fail

; LINUX-X64: test10d:
; LINUX-X64: mov{{l|q}} %fs:
; LINUX-X64: callq __stack_chk_fail

; LINUX-KERNEL-X64: test10d:
; LINUX-KERNEL-X64: mov{{l|q}} %gs:
; LINUX-KERNEL-X64: callq __stack_chk_fail

; DARWIN-X64: test10d:
; DARWIN-X64: mov{{l|q}} ___stack_chk_guard
; DARWIN-X64: callq ___stack_chk_fail
  %x = alloca double, align 8
  %call = call double @testi_aux() nounwind
  store double %call, double* %x, align 8
  %cmp = fcmp ogt double %call, 3.140000e+00
  br i1 %cmp, label %if.then, label %if.else

if.then:                                          ; preds = %entry
  %call1 = call double @testi_aux() nounwind
  store double %call1, double* %x, align 8
  br label %if.end4

if.else:                                          ; preds = %entry
  %cmp2 = fcmp ogt double %call, 1.000000e+00
  br i1 %cmp2, label %if.then3, label %if.end4

if.then3:                                         ; preds = %if.else
  br label %if.end4

if.end4:                                          ; preds = %if.else, %if.then3, %if.then
  %y.0 = phi double* [ null, %if.then ], [ %x, %if.then3 ], [ null, %if.else ]
  %call5 = call i32 (i8*, ...)* @printf(i8* getelementptr inbounds ([4 x i8]* @.str, i64 0, i64 0), double* %y.0) nounwind
  ret void
}

; test11a: Addr-of struct element. (GEP followed by store).
;          no ssp attribute
; Requires no protector.
define void @test11a() nounwind uwtable {
entry:
; LINUX-I386: test11a:
; LINUX-I386-NOT: calll __stack_chk_fail
; LINUX-I386: .cfi_endproc

; LINUX-X64: test11a:
; LINUX-X64-NOT: callq __stack_chk_fail
; LINUX-X64: .cfi_endproc

; LINUX-KERNEL-X64: test11a:
; LINUX-KERNEL-X64-NOT: callq __stack_chk_fail
; LINUX-KERNEL-X64: .cfi_endproc

; DARWIN-X64: test11a:
; DARWIN-X64-NOT: callq ___stack_chk_fail
; DARWIN-X64: .cfi_endproc
  %c = alloca %struct.pair, align 4
  %b = alloca i32*, align 8
  %y = getelementptr inbounds %struct.pair* %c, i32 0, i32 1
  store i32* %y, i32** %b, align 8
  %0 = load i32** %b, align 8
  %call = call i32 (i8*, ...)* @printf(i8* getelementptr inbounds ([4 x i8]* @.str, i32 0, i32 0), i32* %0)
  ret void
}

; test11b: Addr-of struct element. (GEP followed by store).
;          ssp attribute
; Requires no protector.
define void @test11b() nounwind uwtable ssp {
entry:
; LINUX-I386: test11b:
; LINUX-I386-NOT: calll __stack_chk_fail
; LINUX-I386: .cfi_endproc

; LINUX-X64: test11b:
; LINUX-X64-NOT: callq __stack_chk_fail
; LINUX-X64: .cfi_endproc

; LINUX-KERNEL-X64: test11b:
; LINUX-KERNEL-X64-NOT: callq __stack_chk_fail
; LINUX-KERNEL-X64: .cfi_endproc

; DARWIN-X64: test11b:
; DARWIN-X64-NOT: callq ___stack_chk_fail
; DARWIN-X64: .cfi_endproc
  %c = alloca %struct.pair, align 4
  %b = alloca i32*, align 8
  %y = getelementptr inbounds %struct.pair* %c, i32 0, i32 1
  store i32* %y, i32** %b, align 8
  %0 = load i32** %b, align 8
  %call = call i32 (i8*, ...)* @printf(i8* getelementptr inbounds ([4 x i8]* @.str, i32 0, i32 0), i32* %0)
  ret void
}

; test11c: Addr-of struct element. (GEP followed by store).
;          sspstrong attribute
; Requires protector.
define void @test11c() nounwind uwtable sspstrong {
entry:
; LINUX-I386: test11c:
; LINUX-I386: mov{{l|q}} %gs:
; LINUX-I386: calll __stack_chk_fail

; LINUX-X64: test11c:
; LINUX-X64: mov{{l|q}} %fs:
; LINUX-X64: callq __stack_chk_fail

; LINUX-KERNEL-X64: test11c:
; LINUX-KERNEL-X64: mov{{l|q}} %gs:
; LINUX-KERNEL-X64: callq __stack_chk_fail

; DARWIN-X64: test11c:
; DARWIN-X64: mov{{l|q}} ___stack_chk_guard
; DARWIN-X64: callq ___stack_chk_fail
  %c = alloca %struct.pair, align 4
  %b = alloca i32*, align 8
  %y = getelementptr inbounds %struct.pair* %c, i32 0, i32 1
  store i32* %y, i32** %b, align 8
  %0 = load i32** %b, align 8
  %call = call i32 (i8*, ...)* @printf(i8* getelementptr inbounds ([4 x i8]* @.str, i32 0, i32 0), i32* %0)
  ret void
}

; test11d: Addr-of struct element. (GEP followed by store).
;          sspreq attribute
; Requires protector.
define void @test11d() nounwind uwtable sspreq {
entry:
; LINUX-I386: test11d:
; LINUX-I386: mov{{l|q}} %gs:
; LINUX-I386: calll __stack_chk_fail

; LINUX-X64: test11d:
; LINUX-X64: mov{{l|q}} %fs:
; LINUX-X64: callq __stack_chk_fail

; LINUX-KERNEL-X64: test11d:
; LINUX-KERNEL-X64: mov{{l|q}} %gs:
; LINUX-KERNEL-X64: callq __stack_chk_fail

; DARWIN-X64: test11d:
; DARWIN-X64: mov{{l|q}} ___stack_chk_guard
; DARWIN-X64: callq ___stack_chk_fail
  %c = alloca %struct.pair, align 4
  %b = alloca i32*, align 8
  %y = getelementptr inbounds %struct.pair* %c, i32 0, i32 1
  store i32* %y, i32** %b, align 8
  %0 = load i32** %b, align 8
  %call = call i32 (i8*, ...)* @printf(i8* getelementptr inbounds ([4 x i8]* @.str, i32 0, i32 0), i32* %0)
  ret void
}

; test12a: Addr-of struct element, GEP followed by ptrtoint.
;          no ssp attribute
; Requires no protector.
define void @test12a() nounwind uwtable {
entry:
; LINUX-I386: test12a:
; LINUX-I386-NOT: calll __stack_chk_fail
; LINUX-I386: .cfi_endproc

; LINUX-X64: test12a:
; LINUX-X64-NOT: callq __stack_chk_fail
; LINUX-X64: .cfi_endproc

; LINUX-KERNEL-X64: test12a:
; LINUX-KERNEL-X64-NOT: callq __stack_chk_fail
; LINUX-KERNEL-X64: .cfi_endproc

; DARWIN-X64: test12a:
; DARWIN-X64-NOT: callq ___stack_chk_fail
; DARWIN-X64: .cfi_endproc
  %c = alloca %struct.pair, align 4
  %b = alloca i32*, align 8
  %y = getelementptr inbounds %struct.pair* %c, i32 0, i32 1
  %0 = ptrtoint i32* %y to i64
  %call = call i32 (i8*, ...)* @printf(i8* getelementptr inbounds ([4 x i8]* @.str, i32 0, i32 0), i64 %0)
  ret void
}

; test12b: Addr-of struct element, GEP followed by ptrtoint.
;          ssp attribute
; Requires no protector.
define void @test12b() nounwind uwtable ssp {
entry:
; LINUX-I386: test12b:
; LINUX-I386-NOT: calll __stack_chk_fail
; LINUX-I386: .cfi_endproc

; LINUX-X64: test12b:
; LINUX-X64-NOT: callq __stack_chk_fail
; LINUX-X64: .cfi_endproc

; LINUX-KERNEL-X64: test12b:
; LINUX-KERNEL-X64-NOT: callq __stack_chk_fail
; LINUX-KERNEL-X64: .cfi_endproc

; DARWIN-X64: test12b:
; DARWIN-X64-NOT: callq ___stack_chk_fail
; DARWIN-X64: .cfi_endproc
  %c = alloca %struct.pair, align 4
  %b = alloca i32*, align 8
  %y = getelementptr inbounds %struct.pair* %c, i32 0, i32 1
  %0 = ptrtoint i32* %y to i64
  %call = call i32 (i8*, ...)* @printf(i8* getelementptr inbounds ([4 x i8]* @.str, i32 0, i32 0), i64 %0)
  ret void
}

; test12c: Addr-of struct element, GEP followed by ptrtoint.
;          sspstrong attribute
; Requires protector.
define void @test12c() nounwind uwtable sspstrong {
entry:
; LINUX-I386: test12c:
; LINUX-I386: mov{{l|q}} %gs:
; LINUX-I386: calll __stack_chk_fail

; LINUX-X64: test12c:
; LINUX-X64: mov{{l|q}} %fs:
; LINUX-X64: callq __stack_chk_fail

; LINUX-KERNEL-X64: test12c:
; LINUX-KERNEL-X64: mov{{l|q}} %gs:
; LINUX-KERNEL-X64: callq __stack_chk_fail

; DARWIN-X64: test12c:
; DARWIN-X64: mov{{l|q}} ___stack_chk_guard
; DARWIN-X64: callq ___stack_chk_fail
  %c = alloca %struct.pair, align 4
  %b = alloca i32*, align 8
  %y = getelementptr inbounds %struct.pair* %c, i32 0, i32 1
  %0 = ptrtoint i32* %y to i64
  %call = call i32 (i8*, ...)* @printf(i8* getelementptr inbounds ([4 x i8]* @.str, i32 0, i32 0), i64 %0)
  ret void
}

; test12d: Addr-of struct element, GEP followed by ptrtoint.
;          sspreq attribute
; Requires protector.
define void @test12d() nounwind uwtable sspreq {
entry:
; LINUX-I386: test12d:
; LINUX-I386: mov{{l|q}} %gs:
; LINUX-I386: calll __stack_chk_fail

; LINUX-X64: test12d:
; LINUX-X64: mov{{l|q}} %fs:
; LINUX-X64: callq __stack_chk_fail

; LINUX-KERNEL-X64: test12d:
; LINUX-KERNEL-X64: mov{{l|q}} %gs:
; LINUX-KERNEL-X64: callq __stack_chk_fail

; DARWIN-X64: test12d:
; DARWIN-X64: mov{{l|q}} ___stack_chk_guard
; DARWIN-X64: callq ___stack_chk_fail
  %c = alloca %struct.pair, align 4
  %b = alloca i32*, align 8
  %y = getelementptr inbounds %struct.pair* %c, i32 0, i32 1
  %0 = ptrtoint i32* %y to i64
  %call = call i32 (i8*, ...)* @printf(i8* getelementptr inbounds ([4 x i8]* @.str, i32 0, i32 0), i64 %0)
  ret void
}

; test13a: Addr-of struct element, GEP followed by callinst.
;          no ssp attribute
; Requires no protector.
define void @test13a() nounwind uwtable {
entry:
; LINUX-I386: test13a:
; LINUX-I386-NOT: calll __stack_chk_fail
; LINUX-I386: .cfi_endproc

; LINUX-X64: test13a:
; LINUX-X64-NOT: callq __stack_chk_fail
; LINUX-X64: .cfi_endproc

; LINUX-KERNEL-X64: test13a:
; LINUX-KERNEL-X64-NOT: callq __stack_chk_fail
; LINUX-KERNEL-X64: .cfi_endproc

; DARWIN-X64: test13a:
; DARWIN-X64-NOT: callq ___stack_chk_fail
; DARWIN-X64: .cfi_endproc
  %c = alloca %struct.pair, align 4
  %y = getelementptr inbounds %struct.pair* %c, i64 0, i32 1
  %call = call i32 (i8*, ...)* @printf(i8* getelementptr inbounds ([4 x i8]* @.str, i64 0, i64 0), i32* %y) nounwind
  ret void
}

; test13b: Addr-of struct element, GEP followed by callinst.
;          ssp attribute
; Requires no protector.
define void @test13b() nounwind uwtable ssp {
entry:
; LINUX-I386: test13b:
; LINUX-I386-NOT: calll __stack_chk_fail
; LINUX-I386: .cfi_endproc

; LINUX-X64: test13b:
; LINUX-X64-NOT: callq __stack_chk_fail
; LINUX-X64: .cfi_endproc

; LINUX-KERNEL-X64: test13b:
; LINUX-KERNEL-X64-NOT: callq __stack_chk_fail
; LINUX-KERNEL-X64: .cfi_endproc

; DARWIN-X64: test13b:
; DARWIN-X64-NOT: callq ___stack_chk_fail
; DARWIN-X64: .cfi_endproc
  %c = alloca %struct.pair, align 4
  %y = getelementptr inbounds %struct.pair* %c, i64 0, i32 1
  %call = call i32 (i8*, ...)* @printf(i8* getelementptr inbounds ([4 x i8]* @.str, i64 0, i64 0), i32* %y) nounwind
  ret void
}

; test13c: Addr-of struct element, GEP followed by callinst.
;          sspstrong attribute
; Requires protector.
define void @test13c() nounwind uwtable sspstrong {
entry:
; LINUX-I386: test13c:
; LINUX-I386: mov{{l|q}} %gs:
; LINUX-I386: calll __stack_chk_fail

; LINUX-X64: test13c:
; LINUX-X64: mov{{l|q}} %fs:
; LINUX-X64: callq __stack_chk_fail

; LINUX-KERNEL-X64: test13c:
; LINUX-KERNEL-X64: mov{{l|q}} %gs:
; LINUX-KERNEL-X64: callq __stack_chk_fail

; DARWIN-X64: test13c:
; DARWIN-X64: mov{{l|q}} ___stack_chk_guard
; DARWIN-X64: callq ___stack_chk_fail
  %c = alloca %struct.pair, align 4
  %y = getelementptr inbounds %struct.pair* %c, i64 0, i32 1
  %call = call i32 (i8*, ...)* @printf(i8* getelementptr inbounds ([4 x i8]* @.str, i64 0, i64 0), i32* %y) nounwind
  ret void
}

; test13d: Addr-of struct element, GEP followed by callinst.
;          sspreq attribute
; Requires protector.
define void @test13d() nounwind uwtable sspreq {
entry:
; LINUX-I386: test13d:
; LINUX-I386: mov{{l|q}} %gs:
; LINUX-I386: calll __stack_chk_fail

; LINUX-X64: test13d:
; LINUX-X64: mov{{l|q}} %fs:
; LINUX-X64: callq __stack_chk_fail

; LINUX-KERNEL-X64: test13d:
; LINUX-KERNEL-X64: mov{{l|q}} %gs:
; LINUX-KERNEL-X64: callq __stack_chk_fail

; DARWIN-X64: test13d:
; DARWIN-X64: mov{{l|q}} ___stack_chk_guard
; DARWIN-X64: callq ___stack_chk_fail
  %c = alloca %struct.pair, align 4
  %y = getelementptr inbounds %struct.pair* %c, i64 0, i32 1
  %call = call i32 (i8*, ...)* @printf(i8* getelementptr inbounds ([4 x i8]* @.str, i64 0, i64 0), i32* %y) nounwind
  ret void
}

; test14a: Addr-of a local, optimized into a GEP (e.g., &a - 12)
;          no ssp attribute
; Requires no protector.
define void @test14a() nounwind uwtable {
entry:
; LINUX-I386: test14a:
; LINUX-I386-NOT: calll __stack_chk_fail
; LINUX-I386: .cfi_endproc

; LINUX-X64: test14a:
; LINUX-X64-NOT: callq __stack_chk_fail
; LINUX-X64: .cfi_endproc

; LINUX-KERNEL-X64: test14a:
; LINUX-KERNEL-X64-NOT: callq __stack_chk_fail
; LINUX-KERNEL-X64: .cfi_endproc

; DARWIN-X64: test14a:
; DARWIN-X64-NOT: callq ___stack_chk_fail
; DARWIN-X64: .cfi_endproc
  %a = alloca i32, align 4
  %add.ptr5 = getelementptr inbounds i32* %a, i64 -12
  %call = call i32 (i8*, ...)* @printf(i8* getelementptr inbounds ([4 x i8]* @.str, i64 0, i64 0), i32* %add.ptr5) nounwind
  ret void
}

; test14b: Addr-of a local, optimized into a GEP (e.g., &a - 12)
;          ssp attribute
; Requires no protector.
define void @test14b() nounwind uwtable ssp {
entry:
; LINUX-I386: test14b:
; LINUX-I386-NOT: calll __stack_chk_fail
; LINUX-I386: .cfi_endproc

; LINUX-X64: test14b:
; LINUX-X64-NOT: callq __stack_chk_fail
; LINUX-X64: .cfi_endproc

; LINUX-KERNEL-X64: test14b:
; LINUX-KERNEL-X64-NOT: callq __stack_chk_fail
; LINUX-KERNEL-X64: .cfi_endproc

; DARWIN-X64: test14b:
; DARWIN-X64-NOT: callq ___stack_chk_fail
; DARWIN-X64: .cfi_endproc
  %a = alloca i32, align 4
  %add.ptr5 = getelementptr inbounds i32* %a, i64 -12
  %call = call i32 (i8*, ...)* @printf(i8* getelementptr inbounds ([4 x i8]* @.str, i64 0, i64 0), i32* %add.ptr5) nounwind
  ret void
}

; test14c: Addr-of a local, optimized into a GEP (e.g., &a - 12)
;          sspstrong attribute
; Requires protector.
define void @test14c() nounwind uwtable sspstrong {
entry:
; LINUX-I386: test14c:
; LINUX-I386: mov{{l|q}} %gs:
; LINUX-I386: calll __stack_chk_fail

; LINUX-X64: test14c:
; LINUX-X64: mov{{l|q}} %fs:
; LINUX-X64: callq __stack_chk_fail

; LINUX-KERNEL-X64: test14c:
; LINUX-KERNEL-X64: mov{{l|q}} %gs:
; LINUX-KERNEL-X64: callq __stack_chk_fail

; DARWIN-X64: test14c:
; DARWIN-X64: mov{{l|q}} ___stack_chk_guard
; DARWIN-X64: callq ___stack_chk_fail
  %a = alloca i32, align 4
  %add.ptr5 = getelementptr inbounds i32* %a, i64 -12
  %call = call i32 (i8*, ...)* @printf(i8* getelementptr inbounds ([4 x i8]* @.str, i64 0, i64 0), i32* %add.ptr5) nounwind
  ret void
}

; test14d: Addr-of a local, optimized into a GEP (e.g., &a - 12)
;          sspreq  attribute
; Requires protector.
define void @test14d() nounwind uwtable sspreq {
entry:
; LINUX-I386: test14d:
; LINUX-I386: mov{{l|q}} %gs:
; LINUX-I386: calll __stack_chk_fail

; LINUX-X64: test14d:
; LINUX-X64: mov{{l|q}} %fs:
; LINUX-X64: callq __stack_chk_fail

; LINUX-KERNEL-X64: test14d:
; LINUX-KERNEL-X64: mov{{l|q}} %gs:
; LINUX-KERNEL-X64: callq __stack_chk_fail

; DARWIN-X64: test14d:
; DARWIN-X64: mov{{l|q}} ___stack_chk_guard
; DARWIN-X64: callq ___stack_chk_fail
  %a = alloca i32, align 4
  %add.ptr5 = getelementptr inbounds i32* %a, i64 -12
  %call = call i32 (i8*, ...)* @printf(i8* getelementptr inbounds ([4 x i8]* @.str, i64 0, i64 0), i32* %add.ptr5) nounwind
  ret void
}

; test15a: Addr-of a local cast to a ptr of a different type
;           (e.g., int a; ... ; float *b = &a;)
;          no ssp attribute
; Requires no protector.
define void @test15a() nounwind uwtable {
entry:
; LINUX-I386: test15a:
; LINUX-I386-NOT: calll __stack_chk_fail
; LINUX-I386: .cfi_endproc

; LINUX-X64: test15a:
; LINUX-X64-NOT: callq __stack_chk_fail
; LINUX-X64: .cfi_endproc

; LINUX-KERNEL-X64: test15a:
; LINUX-KERNEL-X64-NOT: callq __stack_chk_fail
; LINUX-KERNEL-X64: .cfi_endproc

; DARWIN-X64: test15a:
; DARWIN-X64-NOT: callq ___stack_chk_fail
; DARWIN-X64: .cfi_endproc
  %a = alloca i32, align 4
  %b = alloca float*, align 8
  store i32 0, i32* %a, align 4
  %0 = bitcast i32* %a to float*
  store float* %0, float** %b, align 8
  %1 = load float** %b, align 8
  %call = call i32 (i8*, ...)* @printf(i8* getelementptr inbounds ([4 x i8]* @.str, i32 0, i32 0), float* %1)
  ret void
}

; test15b: Addr-of a local cast to a ptr of a different type
;           (e.g., int a; ... ; float *b = &a;)
;          ssp attribute
; Requires no protector.
define void @test15b() nounwind uwtable ssp {
entry:
; LINUX-I386: test15b:
; LINUX-I386-NOT: calll __stack_chk_fail
; LINUX-I386: .cfi_endproc

; LINUX-X64: test15b:
; LINUX-X64-NOT: callq __stack_chk_fail
; LINUX-X64: .cfi_endproc

; LINUX-KERNEL-X64: test15b:
; LINUX-KERNEL-X64-NOT: callq __stack_chk_fail
; LINUX-KERNEL-X64: .cfi_endproc

; DARWIN-X64: test15b:
; DARWIN-X64-NOT: callq ___stack_chk_fail
; DARWIN-X64: .cfi_endproc
  %a = alloca i32, align 4
  %b = alloca float*, align 8
  store i32 0, i32* %a, align 4
  %0 = bitcast i32* %a to float*
  store float* %0, float** %b, align 8
  %1 = load float** %b, align 8
  %call = call i32 (i8*, ...)* @printf(i8* getelementptr inbounds ([4 x i8]* @.str, i32 0, i32 0), float* %1)
  ret void
}

; test15c: Addr-of a local cast to a ptr of a different type
;           (e.g., int a; ... ; float *b = &a;)
;          sspstrong attribute
; Requires protector.
define void @test15c() nounwind uwtable sspstrong {
entry:
; LINUX-I386: test15c:
; LINUX-I386: mov{{l|q}} %gs:
; LINUX-I386: calll __stack_chk_fail

; LINUX-X64: test15c:
; LINUX-X64: mov{{l|q}} %fs:
; LINUX-X64: callq __stack_chk_fail

; LINUX-KERNEL-X64: test15c:
; LINUX-KERNEL-X64: mov{{l|q}} %gs:
; LINUX-KERNEL-X64: callq __stack_chk_fail

; DARWIN-X64: test15c:
; DARWIN-X64: mov{{l|q}} ___stack_chk_guard
; DARWIN-X64: callq ___stack_chk_fail
  %a = alloca i32, align 4
  %b = alloca float*, align 8
  store i32 0, i32* %a, align 4
  %0 = bitcast i32* %a to float*
  store float* %0, float** %b, align 8
  %1 = load float** %b, align 8
  %call = call i32 (i8*, ...)* @printf(i8* getelementptr inbounds ([4 x i8]* @.str, i32 0, i32 0), float* %1)
  ret void
}

; test15d: Addr-of a local cast to a ptr of a different type
;           (e.g., int a; ... ; float *b = &a;)
;          sspreq attribute
; Requires protector.
define void @test15d() nounwind uwtable sspreq {
entry:
; LINUX-I386: test15d:
; LINUX-I386: mov{{l|q}} %gs:
; LINUX-I386: calll __stack_chk_fail

; LINUX-X64: test15d:
; LINUX-X64: mov{{l|q}} %fs:
; LINUX-X64: callq __stack_chk_fail

; LINUX-KERNEL-X64: test15d:
; LINUX-KERNEL-X64: mov{{l|q}} %gs:
; LINUX-KERNEL-X64: callq __stack_chk_fail

; DARWIN-X64: test15d:
; DARWIN-X64: mov{{l|q}} ___stack_chk_guard
; DARWIN-X64: callq ___stack_chk_fail
  %a = alloca i32, align 4
  %b = alloca float*, align 8
  store i32 0, i32* %a, align 4
  %0 = bitcast i32* %a to float*
  store float* %0, float** %b, align 8
  %1 = load float** %b, align 8
  %call = call i32 (i8*, ...)* @printf(i8* getelementptr inbounds ([4 x i8]* @.str, i32 0, i32 0), float* %1)
  ret void
}

; test16a: Addr-of a local cast to a ptr of a different type (optimized)
;           (e.g., int a; ... ; float *b = &a;)
;          no ssp attribute
; Requires no protector.
define void @test16a() nounwind uwtable {
entry:
; LINUX-I386: test16a:
; LINUX-I386-NOT: calll __stack_chk_fail
; LINUX-I386: .cfi_endproc

; LINUX-X64: test16a:
; LINUX-X64-NOT: callq __stack_chk_fail
; LINUX-X64: .cfi_endproc

; LINUX-KERNEL-X64: test16a:
; LINUX-KERNEL-X64-NOT: callq __stack_chk_fail
; LINUX-KERNEL-X64: .cfi_endproc

; DARWIN-X64: test16a:
; DARWIN-X64-NOT: callq ___stack_chk_fail
; DARWIN-X64: .cfi_endproc
  %a = alloca i32, align 4
  store i32 0, i32* %a, align 4
  %0 = bitcast i32* %a to float*
  call void @funfloat(float* %0) nounwind
  ret void
}

; test16b: Addr-of a local cast to a ptr of a different type (optimized)
;           (e.g., int a; ... ; float *b = &a;)
;          ssp attribute
; Requires no protector.
define void @test16b() nounwind uwtable ssp {
entry:
; LINUX-I386: test16b:
; LINUX-I386-NOT: calll __stack_chk_fail
; LINUX-I386: .cfi_endproc

; LINUX-X64: test16b:
; LINUX-X64-NOT: callq __stack_chk_fail
; LINUX-X64: .cfi_endproc

; LINUX-KERNEL-X64: test16b:
; LINUX-KERNEL-X64-NOT: callq __stack_chk_fail
; LINUX-KERNEL-X64: .cfi_endproc

; DARWIN-X64: test16b:
; DARWIN-X64-NOT: callq ___stack_chk_fail
; DARWIN-X64: .cfi_endproc
  %a = alloca i32, align 4
  store i32 0, i32* %a, align 4
  %0 = bitcast i32* %a to float*
  call void @funfloat(float* %0) nounwind
  ret void
}

; test16c: Addr-of a local cast to a ptr of a different type (optimized)
;           (e.g., int a; ... ; float *b = &a;)
;          sspstrong attribute
; Requires protector.
define void @test16c() nounwind uwtable sspstrong {
entry:
; LINUX-I386: test16c:
; LINUX-I386: mov{{l|q}} %gs:
; LINUX-I386: calll __stack_chk_fail

; LINUX-X64: test16c:
; LINUX-X64: mov{{l|q}} %fs:
; LINUX-X64: callq __stack_chk_fail

; LINUX-KERNEL-X64: test16c:
; LINUX-KERNEL-X64: mov{{l|q}} %gs:
; LINUX-KERNEL-X64: callq __stack_chk_fail

; DARWIN-X64: test16c:
; DARWIN-X64: mov{{l|q}} ___stack_chk_guard
; DARWIN-X64: callq ___stack_chk_fail
  %a = alloca i32, align 4
  store i32 0, i32* %a, align 4
  %0 = bitcast i32* %a to float*
  call void @funfloat(float* %0) nounwind
  ret void
}

; test16d: Addr-of a local cast to a ptr of a different type (optimized)
;           (e.g., int a; ... ; float *b = &a;)
;          sspreq attribute
; Requires protector.
define void @test16d() nounwind uwtable sspreq {
entry:
; LINUX-I386: test16d:
; LINUX-I386: mov{{l|q}} %gs:
; LINUX-I386: calll __stack_chk_fail

; LINUX-X64: test16d:
; LINUX-X64: mov{{l|q}} %fs:
; LINUX-X64: callq __stack_chk_fail

; LINUX-KERNEL-X64: test16d:
; LINUX-KERNEL-X64: mov{{l|q}} %gs:
; LINUX-KERNEL-X64: callq __stack_chk_fail

; DARWIN-X64: test16d:
; DARWIN-X64: mov{{l|q}} ___stack_chk_guard
; DARWIN-X64: callq ___stack_chk_fail
  %a = alloca i32, align 4
  store i32 0, i32* %a, align 4
  %0 = bitcast i32* %a to float*
  call void @funfloat(float* %0) nounwind
  ret void
}

; test17a: Addr-of a vector nested in a struct
;          no ssp attribute
; Requires no protector.
define void @test17a() nounwind uwtable {
entry:
; LINUX-I386: test17a:
; LINUX-I386-NOT: calll __stack_chk_fail
; LINUX-I386: .cfi_endproc

; LINUX-X64: test17a:
; LINUX-X64-NOT: callq __stack_chk_fail
; LINUX-X64: .cfi_endproc

; LINUX-KERNEL-X64: test17a:
; LINUX-KERNEL-X64-NOT: callq __stack_chk_fail
; LINUX-KERNEL-X64: .cfi_endproc

; DARWIN-X64: test17a:
; DARWIN-X64-NOT: callq ___stack_chk_fail
; DARWIN-X64: .cfi_endproc
  %c = alloca %struct.vec, align 16
  %y = getelementptr inbounds %struct.vec* %c, i64 0, i32 0
  %add.ptr = getelementptr inbounds <4 x i32>* %y, i64 -12
  %call = call i32 (i8*, ...)* @printf(i8* getelementptr inbounds ([4 x i8]* @.str, i64 0, i64 0), <4 x i32>* %add.ptr) nounwind
  ret void
}

; test17b: Addr-of a vector nested in a struct
;          ssp attribute
; Requires no protector.
define void @test17b() nounwind uwtable ssp {
entry:
; LINUX-I386: test17b:
; LINUX-I386-NOT: calll __stack_chk_fail
; LINUX-I386: .cfi_endproc

; LINUX-X64: test17b:
; LINUX-X64-NOT: callq __stack_chk_fail
; LINUX-X64: .cfi_endproc

; LINUX-KERNEL-X64: test17b:
; LINUX-KERNEL-X64-NOT: callq __stack_chk_fail
; LINUX-KERNEL-X64: .cfi_endproc

; DARWIN-X64: test17b:
; DARWIN-X64-NOT: callq ___stack_chk_fail
; DARWIN-X64: .cfi_endproc
  %c = alloca %struct.vec, align 16
  %y = getelementptr inbounds %struct.vec* %c, i64 0, i32 0
  %add.ptr = getelementptr inbounds <4 x i32>* %y, i64 -12
  %call = call i32 (i8*, ...)* @printf(i8* getelementptr inbounds ([4 x i8]* @.str, i64 0, i64 0), <4 x i32>* %add.ptr) nounwind
  ret void
}

; test17c: Addr-of a vector nested in a struct
;          sspstrong attribute
; Requires protector.
define void @test17c() nounwind uwtable sspstrong {
entry:
; LINUX-I386: test17c:
; LINUX-I386: mov{{l|q}} %gs:
; LINUX-I386: calll __stack_chk_fail

; LINUX-X64: test17c:
; LINUX-X64: mov{{l|q}} %fs:
; LINUX-X64: callq __stack_chk_fail

; LINUX-KERNEL-X64: test17c:
; LINUX-KERNEL-X64: mov{{l|q}} %gs:
; LINUX-KERNEL-X64: callq __stack_chk_fail

; DARWIN-X64: test17c:
; DARWIN-X64: mov{{l|q}} ___stack_chk_guard
; DARWIN-X64: callq ___stack_chk_fail
  %c = alloca %struct.vec, align 16
  %y = getelementptr inbounds %struct.vec* %c, i64 0, i32 0
  %add.ptr = getelementptr inbounds <4 x i32>* %y, i64 -12
  %call = call i32 (i8*, ...)* @printf(i8* getelementptr inbounds ([4 x i8]* @.str, i64 0, i64 0), <4 x i32>* %add.ptr) nounwind
  ret void
}

; test17d: Addr-of a vector nested in a struct
;          sspreq attribute
; Requires protector.
define void @test17d() nounwind uwtable sspreq {
entry:
; LINUX-I386: test17d:
; LINUX-I386: mov{{l|q}} %gs:
; LINUX-I386: calll __stack_chk_fail

; LINUX-X64: test17d:
; LINUX-X64: mov{{l|q}} %fs:
; LINUX-X64: callq __stack_chk_fail

; LINUX-KERNEL-X64: test17d:
; LINUX-KERNEL-X64: mov{{l|q}} %gs:
; LINUX-KERNEL-X64: callq __stack_chk_fail

; DARWIN-X64: test17d:
; DARWIN-X64: mov{{l|q}} ___stack_chk_guard
; DARWIN-X64: callq ___stack_chk_fail
  %c = alloca %struct.vec, align 16
  %y = getelementptr inbounds %struct.vec* %c, i64 0, i32 0
  %add.ptr = getelementptr inbounds <4 x i32>* %y, i64 -12
  %call = call i32 (i8*, ...)* @printf(i8* getelementptr inbounds ([4 x i8]* @.str, i64 0, i64 0), <4 x i32>* %add.ptr) nounwind
  ret void
}

; test18a: Addr-of a variable passed into an invoke instruction.
;          no ssp attribute
; Requires no protector.
define i32 @test18a() uwtable {
entry:
; LINUX-I386: test18a:
; LINUX-I386-NOT: calll __stack_chk_fail
; LINUX-I386: .cfi_endproc

; LINUX-X64: test18a:
; LINUX-X64-NOT: callq __stack_chk_fail
; LINUX-X64: .cfi_endproc

; LINUX-KERNEL-X64: test18a:
; LINUX-KERNEL-X64-NOT: callq __stack_chk_fail
; LINUX-KERNEL-X64: .cfi_endproc

; DARWIN-X64: test18a:
; DARWIN-X64-NOT: callq ___stack_chk_fail
; DARWIN-X64: .cfi_endproc
  %a = alloca i32, align 4
  %exn.slot = alloca i8*
  %ehselector.slot = alloca i32
  store i32 0, i32* %a, align 4
  invoke void @_Z3exceptPi(i32* %a)
          to label %invoke.cont unwind label %lpad

invoke.cont:
  ret i32 0

lpad:
  %0 = landingpad { i8*, i32 } personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*)
          catch i8* null
  ret i32 0
}

; test18b: Addr-of a variable passed into an invoke instruction.
;          ssp attribute
; Requires no protector.
define i32 @test18b() uwtable ssp {
entry:
; LINUX-I386: test18b:
; LINUX-I386-NOT: calll __stack_chk_fail
; LINUX-I386: .cfi_endproc

; LINUX-X64: test18b:
; LINUX-X64-NOT: callq __stack_chk_fail
; LINUX-X64: .cfi_endproc

; LINUX-KERNEL-X64: test18b:
; LINUX-KERNEL-X64-NOT: callq __stack_chk_fail
; LINUX-KERNEL-X64: .cfi_endproc

; DARWIN-X64: test18b:
; DARWIN-X64-NOT: callq ___stack_chk_fail
; DARWIN-X64: .cfi_endproc
  %a = alloca i32, align 4
  %exn.slot = alloca i8*
  %ehselector.slot = alloca i32
  store i32 0, i32* %a, align 4
  invoke void @_Z3exceptPi(i32* %a)
          to label %invoke.cont unwind label %lpad

invoke.cont:
  ret i32 0

lpad:
  %0 = landingpad { i8*, i32 } personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*)
          catch i8* null
  ret i32 0
}

; test18c: Addr-of a variable passed into an invoke instruction.
;          sspstrong attribute
; Requires protector.
define i32 @test18c() uwtable sspstrong {
entry:
; LINUX-I386: test18c:
; LINUX-I386: mov{{l|q}} %gs:
; LINUX-I386: calll __stack_chk_fail

; LINUX-X64: test18c:
; LINUX-X64: mov{{l|q}} %fs:
; LINUX-X64: callq __stack_chk_fail

; LINUX-KERNEL-X64: test18c:
; LINUX-KERNEL-X64: mov{{l|q}} %gs:
; LINUX-KERNEL-X64: callq __stack_chk_fail

; DARWIN-X64: test18c:
; DARWIN-X64: mov{{l|q}} ___stack_chk_guard
; DARWIN-X64: callq ___stack_chk_fail
  %a = alloca i32, align 4
  %exn.slot = alloca i8*
  %ehselector.slot = alloca i32
  store i32 0, i32* %a, align 4
  invoke void @_Z3exceptPi(i32* %a)
          to label %invoke.cont unwind label %lpad

invoke.cont:
  ret i32 0

lpad:
  %0 = landingpad { i8*, i32 } personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*)
          catch i8* null
  ret i32 0
}

; test18d: Addr-of a variable passed into an invoke instruction.
;          sspreq attribute
; Requires protector.
define i32 @test18d() uwtable sspreq {
entry:
; LINUX-I386: test18d:
; LINUX-I386: mov{{l|q}} %gs:
; LINUX-I386: calll __stack_chk_fail

; LINUX-X64: test18d:
; LINUX-X64: mov{{l|q}} %fs:
; LINUX-X64: callq __stack_chk_fail

; LINUX-KERNEL-X64: test18d:
; LINUX-KERNEL-X64: mov{{l|q}} %gs:
; LINUX-KERNEL-X64: callq __stack_chk_fail

; DARWIN-X64: test18d:
; DARWIN-X64: mov{{l|q}} ___stack_chk_guard
; DARWIN-X64: callq ___stack_chk_fail
  %a = alloca i32, align 4
  %exn.slot = alloca i8*
  %ehselector.slot = alloca i32
  store i32 0, i32* %a, align 4
  invoke void @_Z3exceptPi(i32* %a)
          to label %invoke.cont unwind label %lpad

invoke.cont:
  ret i32 0

lpad:
  %0 = landingpad { i8*, i32 } personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*)
          catch i8* null
  ret i32 0
}

; test19a: Addr-of a struct element passed into an invoke instruction.
;           (GEP followed by an invoke)
;          no ssp attribute
; Requires no protector.
define i32 @test19a() uwtable {
entry:
; LINUX-I386: test19a:
; LINUX-I386-NOT: calll __stack_chk_fail
; LINUX-I386: .cfi_endproc

; LINUX-X64: test19a:
; LINUX-X64-NOT: callq __stack_chk_fail
; LINUX-X64: .cfi_endproc

; LINUX-KERNEL-X64: test19a:
; LINUX-KERNEL-X64-NOT: callq __stack_chk_fail
; LINUX-KERNEL-X64: .cfi_endproc

; DARWIN-X64: test19a:
; DARWIN-X64-NOT: callq ___stack_chk_fail
; DARWIN-X64: .cfi_endproc
  %c = alloca %struct.pair, align 4
  %exn.slot = alloca i8*
  %ehselector.slot = alloca i32
  %a = getelementptr inbounds %struct.pair* %c, i32 0, i32 0
  store i32 0, i32* %a, align 4
  %a1 = getelementptr inbounds %struct.pair* %c, i32 0, i32 0
  invoke void @_Z3exceptPi(i32* %a1)
          to label %invoke.cont unwind label %lpad

invoke.cont:
  ret i32 0

lpad:
  %0 = landingpad { i8*, i32 } personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*)
          catch i8* null
  ret i32 0
}

; test19b: Addr-of a struct element passed into an invoke instruction.
;           (GEP followed by an invoke)
;          ssp attribute
; Requires no protector.
define i32 @test19b() uwtable ssp {
entry:
; LINUX-I386: test19b:
; LINUX-I386-NOT: calll __stack_chk_fail
; LINUX-I386: .cfi_endproc

; LINUX-X64: test19b:
; LINUX-X64-NOT: callq __stack_chk_fail
; LINUX-X64: .cfi_endproc

; LINUX-KERNEL-X64: test19b:
; LINUX-KERNEL-X64-NOT: callq __stack_chk_fail
; LINUX-KERNEL-X64: .cfi_endproc

; DARWIN-X64: test19b:
; DARWIN-X64-NOT: callq ___stack_chk_fail
; DARWIN-X64: .cfi_endproc
  %c = alloca %struct.pair, align 4
  %exn.slot = alloca i8*
  %ehselector.slot = alloca i32
  %a = getelementptr inbounds %struct.pair* %c, i32 0, i32 0
  store i32 0, i32* %a, align 4
  %a1 = getelementptr inbounds %struct.pair* %c, i32 0, i32 0
  invoke void @_Z3exceptPi(i32* %a1)
          to label %invoke.cont unwind label %lpad

invoke.cont:
  ret i32 0

lpad:
  %0 = landingpad { i8*, i32 } personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*)
          catch i8* null
  ret i32 0
}

; test19c: Addr-of a struct element passed into an invoke instruction.
;           (GEP followed by an invoke)
;          sspstrong attribute
; Requires protector.
define i32 @test19c() uwtable sspstrong {
entry:
; LINUX-I386: test19c:
; LINUX-I386: mov{{l|q}} %gs:
; LINUX-I386: calll __stack_chk_fail

; LINUX-X64: test19c:
; LINUX-X64: mov{{l|q}} %fs:
; LINUX-X64: callq __stack_chk_fail

; LINUX-KERNEL-X64: test19c:
; LINUX-KERNEL-X64: mov{{l|q}} %gs:
; LINUX-KERNEL-X64: callq __stack_chk_fail

; DARWIN-X64: test19c:
; DARWIN-X64: mov{{l|q}} ___stack_chk_guard
; DARWIN-X64: callq ___stack_chk_fail
  %c = alloca %struct.pair, align 4
  %exn.slot = alloca i8*
  %ehselector.slot = alloca i32
  %a = getelementptr inbounds %struct.pair* %c, i32 0, i32 0
  store i32 0, i32* %a, align 4
  %a1 = getelementptr inbounds %struct.pair* %c, i32 0, i32 0
  invoke void @_Z3exceptPi(i32* %a1)
          to label %invoke.cont unwind label %lpad

invoke.cont:
  ret i32 0

lpad:
  %0 = landingpad { i8*, i32 } personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*)
          catch i8* null
  ret i32 0
}

; test19d: Addr-of a struct element passed into an invoke instruction.
;           (GEP followed by an invoke)
;          sspreq attribute
; Requires protector.
define i32 @test19d() uwtable sspreq {
entry:
; LINUX-I386: test19d:
; LINUX-I386: mov{{l|q}} %gs:
; LINUX-I386: calll __stack_chk_fail

; LINUX-X64: test19d:
; LINUX-X64: mov{{l|q}} %fs:
; LINUX-X64: callq __stack_chk_fail

; LINUX-KERNEL-X64: test19d:
; LINUX-KERNEL-X64: mov{{l|q}} %gs:
; LINUX-KERNEL-X64: callq __stack_chk_fail

; DARWIN-X64: test19d:
; DARWIN-X64: mov{{l|q}} ___stack_chk_guard
; DARWIN-X64: callq ___stack_chk_fail
  %c = alloca %struct.pair, align 4
  %exn.slot = alloca i8*
  %ehselector.slot = alloca i32
  %a = getelementptr inbounds %struct.pair* %c, i32 0, i32 0
  store i32 0, i32* %a, align 4
  %a1 = getelementptr inbounds %struct.pair* %c, i32 0, i32 0
  invoke void @_Z3exceptPi(i32* %a1)
          to label %invoke.cont unwind label %lpad

invoke.cont:
  ret i32 0

lpad:
  %0 = landingpad { i8*, i32 } personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*)
          catch i8* null
  ret i32 0
}

; test20a: Addr-of a pointer
;          no ssp attribute
; Requires no protector.
define void @test20a() nounwind uwtable {
entry:
; LINUX-I386: test20a:
; LINUX-I386-NOT: calll __stack_chk_fail
; LINUX-I386: .cfi_endproc

; LINUX-X64: test20a:
; LINUX-X64-NOT: callq __stack_chk_fail
; LINUX-X64: .cfi_endproc

; LINUX-KERNEL-X64: test20a:
; LINUX-KERNEL-X64-NOT: callq __stack_chk_fail
; LINUX-KERNEL-X64: .cfi_endproc

; DARWIN-X64: test20a:
; DARWIN-X64-NOT: callq ___stack_chk_fail
; DARWIN-X64: .cfi_endproc
  %a = alloca i32*, align 8
  %b = alloca i32**, align 8
  %call = call i32* @getp()
  store i32* %call, i32** %a, align 8
  store i32** %a, i32*** %b, align 8
  %0 = load i32*** %b, align 8
  call void @funcall2(i32** %0)
  ret void
}

; test20b: Addr-of a pointer
;          ssp attribute
; Requires no protector.
define void @test20b() nounwind uwtable ssp {
entry:
; LINUX-I386: test20b:
; LINUX-I386-NOT: calll __stack_chk_fail
; LINUX-I386: .cfi_endproc

; LINUX-X64: test20b:
; LINUX-X64-NOT: callq __stack_chk_fail
; LINUX-X64: .cfi_endproc

; LINUX-KERNEL-X64: test20b:
; LINUX-KERNEL-X64-NOT: callq __stack_chk_fail
; LINUX-KERNEL-X64: .cfi_endproc

; DARWIN-X64: test20b:
; DARWIN-X64-NOT: callq ___stack_chk_fail
; DARWIN-X64: .cfi_endproc
  %a = alloca i32*, align 8
  %b = alloca i32**, align 8
  %call = call i32* @getp()
  store i32* %call, i32** %a, align 8
  store i32** %a, i32*** %b, align 8
  %0 = load i32*** %b, align 8
  call void @funcall2(i32** %0)
  ret void
}

; test20c: Addr-of a pointer
;          sspstrong attribute
; Requires protector.
define void @test20c() nounwind uwtable sspstrong {
entry:
; LINUX-I386: test20c:
; LINUX-I386: mov{{l|q}} %gs:
; LINUX-I386: calll __stack_chk_fail

; LINUX-X64: test20c:
; LINUX-X64: mov{{l|q}} %fs:
; LINUX-X64: callq __stack_chk_fail

; LINUX-KERNEL-X64: test20c:
; LINUX-KERNEL-X64: mov{{l|q}} %gs:
; LINUX-KERNEL-X64: callq __stack_chk_fail

; DARWIN-X64: test20c:
; DARWIN-X64: mov{{l|q}} ___stack_chk_guard
; DARWIN-X64: callq ___stack_chk_fail
  %a = alloca i32*, align 8
  %b = alloca i32**, align 8
  %call = call i32* @getp()
  store i32* %call, i32** %a, align 8
  store i32** %a, i32*** %b, align 8
  %0 = load i32*** %b, align 8
  call void @funcall2(i32** %0)
  ret void
}

; test20d: Addr-of a pointer
;          sspreq attribute
; Requires protector.
define void @test20d() nounwind uwtable sspreq {
entry:
; LINUX-I386: test20d:
; LINUX-I386: mov{{l|q}} %gs:
; LINUX-I386: calll __stack_chk_fail

; LINUX-X64: test20d:
; LINUX-X64: mov{{l|q}} %fs:
; LINUX-X64: callq __stack_chk_fail

; LINUX-KERNEL-X64: test20d:
; LINUX-KERNEL-X64: mov{{l|q}} %gs:
; LINUX-KERNEL-X64: callq __stack_chk_fail

; DARWIN-X64: test20d:
; DARWIN-X64: mov{{l|q}} ___stack_chk_guard
; DARWIN-X64: callq ___stack_chk_fail
  %a = alloca i32*, align 8
  %b = alloca i32**, align 8
  %call = call i32* @getp()
  store i32* %call, i32** %a, align 8
  store i32** %a, i32*** %b, align 8
  %0 = load i32*** %b, align 8
  call void @funcall2(i32** %0)
  ret void
}

; test21a: Addr-of a casted pointer
;          no ssp attribute
; Requires no protector.
define void @test21a() nounwind uwtable {
entry:
; LINUX-I386: test21a:
; LINUX-I386-NOT: calll __stack_chk_fail
; LINUX-I386: .cfi_endproc

; LINUX-X64: test21a:
; LINUX-X64-NOT: callq __stack_chk_fail
; LINUX-X64: .cfi_endproc

; LINUX-KERNEL-X64: test21a:
; LINUX-KERNEL-X64-NOT: callq __stack_chk_fail
; LINUX-KERNEL-X64: .cfi_endproc

; DARWIN-X64: test21a:
; DARWIN-X64-NOT: callq ___stack_chk_fail
; DARWIN-X64: .cfi_endproc
  %a = alloca i32*, align 8
  %b = alloca float**, align 8
  %call = call i32* @getp()
  store i32* %call, i32** %a, align 8
  %0 = bitcast i32** %a to float**
  store float** %0, float*** %b, align 8
  %1 = load float*** %b, align 8
  call void @funfloat2(float** %1)
  ret void
}

; test21b: Addr-of a casted pointer
;          ssp attribute
; Requires no protector.
define void @test21b() nounwind uwtable ssp {
entry:
; LINUX-I386: test21b:
; LINUX-I386-NOT: calll __stack_chk_fail
; LINUX-I386: .cfi_endproc

; LINUX-X64: test21b:
; LINUX-X64-NOT: callq __stack_chk_fail
; LINUX-X64: .cfi_endproc

; LINUX-KERNEL-X64: test21b:
; LINUX-KERNEL-X64-NOT: callq __stack_chk_fail
; LINUX-KERNEL-X64: .cfi_endproc

; DARWIN-X64: test21b:
; DARWIN-X64-NOT: callq ___stack_chk_fail
; DARWIN-X64: .cfi_endproc
  %a = alloca i32*, align 8
  %b = alloca float**, align 8
  %call = call i32* @getp()
  store i32* %call, i32** %a, align 8
  %0 = bitcast i32** %a to float**
  store float** %0, float*** %b, align 8
  %1 = load float*** %b, align 8
  call void @funfloat2(float** %1)
  ret void
}

; test21c: Addr-of a casted pointer
;          sspstrong attribute
; Requires protector.
define void @test21c() nounwind uwtable sspstrong {
entry:
; LINUX-I386: test21c:
; LINUX-I386: mov{{l|q}} %gs:
; LINUX-I386: calll __stack_chk_fail

; LINUX-X64: test21c:
; LINUX-X64: mov{{l|q}} %fs:
; LINUX-X64: callq __stack_chk_fail

; LINUX-KERNEL-X64: test21c:
; LINUX-KERNEL-X64: mov{{l|q}} %gs:
; LINUX-KERNEL-X64: callq __stack_chk_fail

; DARWIN-X64: test21c:
; DARWIN-X64: mov{{l|q}} ___stack_chk_guard
; DARWIN-X64: callq ___stack_chk_fail
  %a = alloca i32*, align 8
  %b = alloca float**, align 8
  %call = call i32* @getp()
  store i32* %call, i32** %a, align 8
  %0 = bitcast i32** %a to float**
  store float** %0, float*** %b, align 8
  %1 = load float*** %b, align 8
  call void @funfloat2(float** %1)
  ret void
}

; test21d: Addr-of a casted pointer
;          sspreq attribute
; Requires protector.
define void @test21d() nounwind uwtable sspreq {
entry:
; LINUX-I386: test21d:
; LINUX-I386: mov{{l|q}} %gs:
; LINUX-I386: calll __stack_chk_fail

; LINUX-X64: test21d:
; LINUX-X64: mov{{l|q}} %fs:
; LINUX-X64: callq __stack_chk_fail

; LINUX-KERNEL-X64: test21d:
; LINUX-KERNEL-X64: mov{{l|q}} %gs:
; LINUX-KERNEL-X64: callq __stack_chk_fail

; DARWIN-X64: test21d:
; DARWIN-X64: mov{{l|q}} ___stack_chk_guard
; DARWIN-X64: callq ___stack_chk_fail
  %a = alloca i32*, align 8
  %b = alloca float**, align 8
  %call = call i32* @getp()
  store i32* %call, i32** %a, align 8
  %0 = bitcast i32** %a to float**
  store float** %0, float*** %b, align 8
  %1 = load float*** %b, align 8
  call void @funfloat2(float** %1)
  ret void
}

; test22a: [2 x i8] in a class
;          no ssp attribute
; Requires no protector.
define signext i8 @test22a() nounwind uwtable {
entry:
; LINUX-I386: test22a:
; LINUX-I386-NOT: calll __stack_chk_fail
; LINUX-I386: .cfi_endproc

; LINUX-X64: test22a:
; LINUX-X64-NOT: callq __stack_chk_fail
; LINUX-X64: .cfi_endproc

; LINUX-KERNEL-X64: test22a:
; LINUX-KERNEL-X64-NOT: callq __stack_chk_fail
; LINUX-KERNEL-X64: .cfi_endproc

; DARWIN-X64: test22a:
; DARWIN-X64-NOT: callq ___stack_chk_fail
; DARWIN-X64: .cfi_endproc
  %a = alloca %class.A, align 1
  %array = getelementptr inbounds %class.A* %a, i32 0, i32 0
  %arrayidx = getelementptr inbounds [2 x i8]* %array, i32 0, i64 0
  %0 = load i8* %arrayidx, align 1
  ret i8 %0
}

; test22b: [2 x i8] in a class
;          ssp attribute
; Requires no protector.
define signext i8 @test22b() nounwind uwtable ssp {
entry:
; LINUX-I386: test22b:
; LINUX-I386-NOT: calll __stack_chk_fail
; LINUX-I386: .cfi_endproc

; LINUX-X64: test22b:
; LINUX-X64-NOT: callq __stack_chk_fail
; LINUX-X64: .cfi_endproc

; LINUX-KERNEL-X64: test22b:
; LINUX-KERNEL-X64-NOT: callq __stack_chk_fail
; LINUX-KERNEL-X64: .cfi_endproc

; DARWIN-X64: test22b:
; DARWIN-X64-NOT: callq ___stack_chk_fail
; DARWIN-X64: .cfi_endproc
  %a = alloca %class.A, align 1
  %array = getelementptr inbounds %class.A* %a, i32 0, i32 0
  %arrayidx = getelementptr inbounds [2 x i8]* %array, i32 0, i64 0
  %0 = load i8* %arrayidx, align 1
  ret i8 %0
}

; test22c: [2 x i8] in a class
;          sspstrong attribute
; Requires protector.
define signext i8 @test22c() nounwind uwtable sspstrong {
entry:
; LINUX-I386: test22c:
; LINUX-I386: mov{{l|q}} %gs:
; LINUX-I386: calll __stack_chk_fail

; LINUX-X64: test22c:
; LINUX-X64: mov{{l|q}} %fs:
; LINUX-X64: callq __stack_chk_fail

; LINUX-KERNEL-X64: test22c:
; LINUX-KERNEL-X64: mov{{l|q}} %gs:
; LINUX-KERNEL-X64: callq __stack_chk_fail

; DARWIN-X64: test22c:
; DARWIN-X64: mov{{l|q}} ___stack_chk_guard
; DARWIN-X64: callq ___stack_chk_fail
  %a = alloca %class.A, align 1
  %array = getelementptr inbounds %class.A* %a, i32 0, i32 0
  %arrayidx = getelementptr inbounds [2 x i8]* %array, i32 0, i64 0
  %0 = load i8* %arrayidx, align 1
  ret i8 %0
}

; test22d: [2 x i8] in a class
;          sspreq attribute
; Requires protector.
define signext i8 @test22d() nounwind uwtable sspreq {
entry:
; LINUX-I386: test22d:
; LINUX-I386: mov{{l|q}} %gs:
; LINUX-I386: calll __stack_chk_fail

; LINUX-X64: test22d:
; LINUX-X64: mov{{l|q}} %fs:
; LINUX-X64: callq __stack_chk_fail

; LINUX-KERNEL-X64: test22d:
; LINUX-KERNEL-X64: mov{{l|q}} %gs:
; LINUX-KERNEL-X64: callq __stack_chk_fail

; DARWIN-X64: test22d:
; DARWIN-X64: mov{{l|q}} ___stack_chk_guard
; DARWIN-X64: callq ___stack_chk_fail
  %a = alloca %class.A, align 1
  %array = getelementptr inbounds %class.A* %a, i32 0, i32 0
  %arrayidx = getelementptr inbounds [2 x i8]* %array, i32 0, i64 0
  %0 = load i8* %arrayidx, align 1
  ret i8 %0
}

; test23a: [2 x i8] nested in several layers of structs and unions
;          no ssp attribute
; Requires no protector.
define signext i8 @test23a() nounwind uwtable {
entry:
; LINUX-I386: test23a:
; LINUX-I386-NOT: calll __stack_chk_fail
; LINUX-I386: .cfi_endproc

; LINUX-X64: test23a:
; LINUX-X64-NOT: callq __stack_chk_fail
; LINUX-X64: .cfi_endproc

; LINUX-KERNEL-X64: test23a:
; LINUX-KERNEL-X64-NOT: callq __stack_chk_fail
; LINUX-KERNEL-X64: .cfi_endproc

; DARWIN-X64: test23a:
; DARWIN-X64-NOT: callq ___stack_chk_fail
; DARWIN-X64: .cfi_endproc
  %x = alloca %struct.deep, align 1
  %b = getelementptr inbounds %struct.deep* %x, i32 0, i32 0
  %c = bitcast %union.anon* %b to %struct.anon*
  %d = getelementptr inbounds %struct.anon* %c, i32 0, i32 0
  %e = getelementptr inbounds %struct.anon.0* %d, i32 0, i32 0
  %array = bitcast %union.anon.1* %e to [2 x i8]*
  %arrayidx = getelementptr inbounds [2 x i8]* %array, i32 0, i64 0
  %0 = load i8* %arrayidx, align 1
  ret i8 %0
}

; test23b: [2 x i8] nested in several layers of structs and unions
;          ssp attribute
; Requires no protector.
define signext i8 @test23b() nounwind uwtable ssp {
entry:
; LINUX-I386: test23b:
; LINUX-I386-NOT: calll __stack_chk_fail
; LINUX-I386: .cfi_endproc

; LINUX-X64: test23b:
; LINUX-X64-NOT: callq __stack_chk_fail
; LINUX-X64: .cfi_endproc

; LINUX-KERNEL-X64: test23b:
; LINUX-KERNEL-X64-NOT: callq __stack_chk_fail
; LINUX-KERNEL-X64: .cfi_endproc

; DARWIN-X64: test23b:
; DARWIN-X64-NOT: callq ___stack_chk_fail
; DARWIN-X64: .cfi_endproc
  %x = alloca %struct.deep, align 1
  %b = getelementptr inbounds %struct.deep* %x, i32 0, i32 0
  %c = bitcast %union.anon* %b to %struct.anon*
  %d = getelementptr inbounds %struct.anon* %c, i32 0, i32 0
  %e = getelementptr inbounds %struct.anon.0* %d, i32 0, i32 0
  %array = bitcast %union.anon.1* %e to [2 x i8]*
  %arrayidx = getelementptr inbounds [2 x i8]* %array, i32 0, i64 0
  %0 = load i8* %arrayidx, align 1
  ret i8 %0
}

; test23c: [2 x i8] nested in several layers of structs and unions
;          sspstrong attribute
; Requires protector.
define signext i8 @test23c() nounwind uwtable sspstrong {
entry:
; LINUX-I386: test23c:
; LINUX-I386: mov{{l|q}} %gs:
; LINUX-I386: calll __stack_chk_fail

; LINUX-X64: test23c:
; LINUX-X64: mov{{l|q}} %fs:
; LINUX-X64: callq __stack_chk_fail

; LINUX-KERNEL-X64: test23c:
; LINUX-KERNEL-X64: mov{{l|q}} %gs:
; LINUX-KERNEL-X64: callq __stack_chk_fail

; DARWIN-X64: test23c:
; DARWIN-X64: mov{{l|q}} ___stack_chk_guard
; DARWIN-X64: callq ___stack_chk_fail
  %x = alloca %struct.deep, align 1
  %b = getelementptr inbounds %struct.deep* %x, i32 0, i32 0
  %c = bitcast %union.anon* %b to %struct.anon*
  %d = getelementptr inbounds %struct.anon* %c, i32 0, i32 0
  %e = getelementptr inbounds %struct.anon.0* %d, i32 0, i32 0
  %array = bitcast %union.anon.1* %e to [2 x i8]*
  %arrayidx = getelementptr inbounds [2 x i8]* %array, i32 0, i64 0
  %0 = load i8* %arrayidx, align 1
  ret i8 %0
}

; test23d: [2 x i8] nested in several layers of structs and unions
;          sspreq attribute
; Requires protector.
define signext i8 @test23d() nounwind uwtable sspreq {
entry:
; LINUX-I386: test23d:
; LINUX-I386: mov{{l|q}} %gs:
; LINUX-I386: calll __stack_chk_fail

; LINUX-X64: test23d:
; LINUX-X64: mov{{l|q}} %fs:
; LINUX-X64: callq __stack_chk_fail

; LINUX-KERNEL-X64: test23d:
; LINUX-KERNEL-X64: mov{{l|q}} %gs:
; LINUX-KERNEL-X64: callq __stack_chk_fail

; DARWIN-X64: test23d:
; DARWIN-X64: mov{{l|q}} ___stack_chk_guard
; DARWIN-X64: callq ___stack_chk_fail
  %x = alloca %struct.deep, align 1
  %b = getelementptr inbounds %struct.deep* %x, i32 0, i32 0
  %c = bitcast %union.anon* %b to %struct.anon*
  %d = getelementptr inbounds %struct.anon* %c, i32 0, i32 0
  %e = getelementptr inbounds %struct.anon.0* %d, i32 0, i32 0
  %array = bitcast %union.anon.1* %e to [2 x i8]*
  %arrayidx = getelementptr inbounds [2 x i8]* %array, i32 0, i64 0
  %0 = load i8* %arrayidx, align 1
  ret i8 %0
}

; test24a: Variable sized alloca
;          no ssp attribute
; Requires no protector.
define void @test24a(i32 %n) nounwind uwtable {
entry:
; LINUX-I386: test24a:
; LINUX-I386-NOT: calll __stack_chk_fail
; LINUX-I386: .cfi_endproc

; LINUX-X64: test24a:
; LINUX-X64-NOT: callq __stack_chk_fail
; LINUX-X64: .cfi_endproc

; LINUX-KERNEL-X64: test24a:
; LINUX-KERNEL-X64-NOT: callq __stack_chk_fail
; LINUX-KERNEL-X64: .cfi_endproc

; DARWIN-X64: test24a:
; DARWIN-X64-NOT: callq ___stack_chk_fail
; DARWIN-X64: .cfi_endproc
  %n.addr = alloca i32, align 4
  %a = alloca i32*, align 8
  store i32 %n, i32* %n.addr, align 4
  %0 = load i32* %n.addr, align 4
  %conv = sext i32 %0 to i64
  %1 = alloca i8, i64 %conv
  %2 = bitcast i8* %1 to i32*
  store i32* %2, i32** %a, align 8
  ret void
}

; test24b: Variable sized alloca
;          ssp attribute
; Requires protector.
define void @test24b(i32 %n) nounwind uwtable ssp {
entry:
; LINUX-I386: test24b:
; LINUX-I386: mov{{l|q}} %gs:
; LINUX-I386: calll __stack_chk_fail

; LINUX-X64: test24b:
; LINUX-X64: mov{{l|q}} %fs:
; LINUX-X64: callq __stack_chk_fail

; LINUX-KERNEL-X64: test24b:
; LINUX-KERNEL-X64: mov{{l|q}} %gs:
; LINUX-KERNEL-X64: callq __stack_chk_fail

; DARWIN-X64: test24b:
; DARWIN-X64: mov{{l|q}} ___stack_chk_guard
; DARWIN-X64: callq ___stack_chk_fail
  %n.addr = alloca i32, align 4
  %a = alloca i32*, align 8
  store i32 %n, i32* %n.addr, align 4
  %0 = load i32* %n.addr, align 4
  %conv = sext i32 %0 to i64
  %1 = alloca i8, i64 %conv
  %2 = bitcast i8* %1 to i32*
  store i32* %2, i32** %a, align 8
  ret void
}

; test24c: Variable sized alloca
;          sspstrong attribute
; Requires protector.
define void @test24c(i32 %n) nounwind uwtable sspstrong {
entry:
; LINUX-I386: test24c:
; LINUX-I386: mov{{l|q}} %gs:
; LINUX-I386: calll __stack_chk_fail

; LINUX-X64: test24c:
; LINUX-X64: mov{{l|q}} %fs:
; LINUX-X64: callq __stack_chk_fail

; LINUX-KERNEL-X64: test24c:
; LINUX-KERNEL-X64: mov{{l|q}} %gs:
; LINUX-KERNEL-X64: callq __stack_chk_fail

; DARWIN-X64: test24c:
; DARWIN-X64: mov{{l|q}} ___stack_chk_guard
; DARWIN-X64: callq ___stack_chk_fail
  %n.addr = alloca i32, align 4
  %a = alloca i32*, align 8
  store i32 %n, i32* %n.addr, align 4
  %0 = load i32* %n.addr, align 4
  %conv = sext i32 %0 to i64
  %1 = alloca i8, i64 %conv
  %2 = bitcast i8* %1 to i32*
  store i32* %2, i32** %a, align 8
  ret void
}

; test24d: Variable sized alloca
;          sspreq attribute
; Requires protector.
define void @test24d(i32 %n) nounwind uwtable sspreq  {
entry:
; LINUX-I386: test24d:
; LINUX-I386: mov{{l|q}} %gs:
; LINUX-I386: calll __stack_chk_fail

; LINUX-X64: test24d:
; LINUX-X64: mov{{l|q}} %fs:
; LINUX-X64: callq __stack_chk_fail

; LINUX-KERNEL-X64: test24d:
; LINUX-KERNEL-X64: mov{{l|q}} %gs:
; LINUX-KERNEL-X64: callq __stack_chk_fail

; DARWIN-X64: test24d:
; DARWIN-X64: mov{{l|q}} ___stack_chk_guard
; DARWIN-X64: callq ___stack_chk_fail
  %n.addr = alloca i32, align 4
  %a = alloca i32*, align 8
  store i32 %n, i32* %n.addr, align 4
  %0 = load i32* %n.addr, align 4
  %conv = sext i32 %0 to i64
  %1 = alloca i8, i64 %conv
  %2 = bitcast i8* %1 to i32*
  store i32* %2, i32** %a, align 8
  ret void
}

; test25a: array of [4 x i32]
;          no ssp attribute
; Requires no protector.
define i32 @test25a() nounwind uwtable {
entry:
; LINUX-I386: test25a:
; LINUX-I386-NOT: calll __stack_chk_fail
; LINUX-I386: .cfi_endproc

; LINUX-X64: test25a:
; LINUX-X64-NOT: callq __stack_chk_fail
; LINUX-X64: .cfi_endproc

; LINUX-KERNEL-X64: test25a:
; LINUX-KERNEL-X64-NOT: callq __stack_chk_fail
; LINUX-KERNEL-X64: .cfi_endproc

; DARWIN-X64: test25a:
; DARWIN-X64-NOT: callq ___stack_chk_fail
; DARWIN-X64: .cfi_endproc
  %a = alloca [4 x i32], align 16
  %arrayidx = getelementptr inbounds [4 x i32]* %a, i32 0, i64 0
  %0 = load i32* %arrayidx, align 4
  ret i32 %0
}

; test25b: array of [4 x i32]
;          ssp attribute
; Requires no protector, except for Darwin which _does_ require a protector.
define i32 @test25b() nounwind uwtable ssp {
entry:
; LINUX-I386: test25b:
; LINUX-I386-NOT: calll __stack_chk_fail
; LINUX-I386: .cfi_endproc

; LINUX-X64: test25b:
; LINUX-X64-NOT: callq __stack_chk_fail
; LINUX-X64: .cfi_endproc

; LINUX-KERNEL-X64: test25b:
; LINUX-KERNEL-X64-NOT: callq __stack_chk_fail
; LINUX-KERNEL-X64: .cfi_endproc

; DARWIN-X64: test25b:
; DARWIN-X64: mov{{l|q}} ___stack_chk_guard
; DARWIN-X64: callq ___stack_chk_fail
  %a = alloca [4 x i32], align 16
  %arrayidx = getelementptr inbounds [4 x i32]* %a, i32 0, i64 0
  %0 = load i32* %arrayidx, align 4
  ret i32 %0
}

; test25c: array of [4 x i32]
;          sspstrong attribute
; Requires protector.
define i32 @test25c() nounwind uwtable sspstrong {
entry:
; LINUX-I386: test25c:
; LINUX-I386: mov{{l|q}} %gs:
; LINUX-I386: calll __stack_chk_fail

; LINUX-X64: test25c:
; LINUX-X64: mov{{l|q}} %fs:
; LINUX-X64: callq __stack_chk_fail

; LINUX-KERNEL-X64: test25c:
; LINUX-KERNEL-X64: mov{{l|q}} %gs:
; LINUX-KERNEL-X64: callq __stack_chk_fail

; DARWIN-X64: test25c:
; DARWIN-X64: mov{{l|q}} ___stack_chk_guard
; DARWIN-X64: callq ___stack_chk_fail
  %a = alloca [4 x i32], align 16
  %arrayidx = getelementptr inbounds [4 x i32]* %a, i32 0, i64 0
  %0 = load i32* %arrayidx, align 4
  ret i32 %0
}

; test25d: array of [4 x i32]
;          sspreq attribute
; Requires protector.
define i32 @test25d() nounwind uwtable sspreq {
entry:
; LINUX-I386: test25d:
; LINUX-I386: mov{{l|q}} %gs:
; LINUX-I386: calll __stack_chk_fail

; LINUX-X64: test25d:
; LINUX-X64: mov{{l|q}} %fs:
; LINUX-X64: callq __stack_chk_fail

; LINUX-KERNEL-X64: test25d:
; LINUX-KERNEL-X64: mov{{l|q}} %gs:
; LINUX-KERNEL-X64: callq __stack_chk_fail

; DARWIN-X64: test25d:
; DARWIN-X64: mov{{l|q}} ___stack_chk_guard
; DARWIN-X64: callq ___stack_chk_fail
  %a = alloca [4 x i32], align 16
  %arrayidx = getelementptr inbounds [4 x i32]* %a, i32 0, i64 0
  %0 = load i32* %arrayidx, align 4
  ret i32 %0
}

; test26: Nested structure, no arrays, no address-of expressions.
;         Verify that the resulting gep-of-gep does not incorrectly trigger
;         a stack protector.
;         ssptrong attribute
; Requires no protector.
define void @test26() nounwind uwtable sspstrong {
entry:
; LINUX-I386: test26:
; LINUX-I386-NOT: calll __stack_chk_fail
; LINUX-I386: .cfi_endproc

; LINUX-X64: test26:
; LINUX-X64-NOT: callq __stack_chk_fail
; LINUX-X64: .cfi_endproc

; LINUX-KERNEL-X64: test26:
; LINUX-KERNEL-X64-NOT: callq __stack_chk_fail
; LINUX-KERNEL-X64: .cfi_endproc

; DARWIN-X64: test26:
; DARWIN-X64-NOT: callq ___stack_chk_fail
; DARWIN-X64: .cfi_endproc
  %c = alloca %struct.nest, align 4
  %b = getelementptr inbounds %struct.nest* %c, i32 0, i32 1
  %_a = getelementptr inbounds %struct.pair* %b, i32 0, i32 0
  %0 = load i32* %_a, align 4
  %call = call i32 (i8*, ...)* @printf(i8* getelementptr inbounds ([4 x i8]* @.str, i32 0, i32 0), i32 %0)
  ret void
}

; test27: Address-of a structure taken in a function with a loop where
;         the alloca is an incoming value to a PHI node and a use of that PHI 
;         node is also an incoming value.
;         Verify that the address-of analysis does not get stuck in infinite
;         recursion when chasing the alloca through the PHI nodes.
; Requires protector.
define i32 @test27(i32 %arg) nounwind uwtable sspstrong {
bb:
; LINUX-I386: test27:
; LINUX-I386: mov{{l|q}} %gs:
; LINUX-I386: calll __stack_chk_fail

; LINUX-X64: test27:
; LINUX-X64: mov{{l|q}} %fs:
; LINUX-X64: callq __stack_chk_fail

; LINUX-KERNEL-X64: test27:
; LINUX-KERNEL-X64: mov{{l|q}} %gs:
; LINUX-KERNEL-X64: callq __stack_chk_fail

; DARWIN-X64: test27:
; DARWIN-X64: mov{{l|q}} ___stack_chk_guard
; DARWIN-X64: callq ___stack_chk_fail
  %tmp = alloca %struct.small*, align 8
  %tmp1 = call i32 (...)* @dummy(%struct.small** %tmp) nounwind
  %tmp2 = load %struct.small** %tmp, align 8
  %tmp3 = ptrtoint %struct.small* %tmp2 to i64
  %tmp4 = trunc i64 %tmp3 to i32
  %tmp5 = icmp sgt i32 %tmp4, 0
  br i1 %tmp5, label %bb6, label %bb21

bb6:                                              ; preds = %bb17, %bb
  %tmp7 = phi %struct.small* [ %tmp19, %bb17 ], [ %tmp2, %bb ]
  %tmp8 = phi i64 [ %tmp20, %bb17 ], [ 1, %bb ]
  %tmp9 = phi i32 [ %tmp14, %bb17 ], [ %tmp1, %bb ]
  %tmp10 = getelementptr inbounds %struct.small* %tmp7, i64 0, i32 0
  %tmp11 = load i8* %tmp10, align 1
  %tmp12 = icmp eq i8 %tmp11, 1
  %tmp13 = add nsw i32 %tmp9, 8
  %tmp14 = select i1 %tmp12, i32 %tmp13, i32 %tmp9
  %tmp15 = trunc i64 %tmp8 to i32
  %tmp16 = icmp eq i32 %tmp15, %tmp4
  br i1 %tmp16, label %bb21, label %bb17

bb17:                                             ; preds = %bb6
  %tmp18 = getelementptr inbounds %struct.small** %tmp, i64 %tmp8
  %tmp19 = load %struct.small** %tmp18, align 8
  %tmp20 = add i64 %tmp8, 1
  br label %bb6

bb21:                                             ; preds = %bb6, %bb
  %tmp22 = phi i32 [ %tmp1, %bb ], [ %tmp14, %bb6 ]
  %tmp23 = call i32 (...)* @dummy(i32 %tmp22) nounwind
  ret i32 undef
}

declare double @testi_aux()
declare i8* @strcpy(i8*, i8*)
declare i32 @printf(i8*, ...)
declare void @funcall(i32*)
declare void @funcall2(i32**)
declare void @funfloat(float*)
declare void @funfloat2(float**)
declare void @_Z3exceptPi(i32*)
declare i32 @__gxx_personality_v0(...)
declare i32* @getp()
declare i32 @dummy(...)
