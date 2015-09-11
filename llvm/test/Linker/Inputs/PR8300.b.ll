%foo = type { [8 x i8] }
%bar = type { [9 x i8] }

@zed = alias void (%foo*), bitcast (void (%bar*)* @xyz to void (%foo*)*)

define void @xyz(%bar* %this) {
entry:
  ret void
}
