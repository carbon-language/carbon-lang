; RUN: %lli %s > /dev/null

@x = thread_local local_unnamed_addr global i32 0

define i32 @main() {
entry:
  store i32 42, i32* @x
  ret i32 0
}

