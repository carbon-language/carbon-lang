; RUN: opt -jump-threading -S < %s

define void @func(i8 zeroext %p_44) nounwind {
entry:
  br i1 false, label %for.cond2, label %if.end50

for.cond2:                                        ; preds = %for.inc46, %lor.end, %entry
  %p_44.addr.1 = phi i8 [ %p_44.addr.1, %lor.end ], [ %p_44, %entry ], [ %p_44.addr.1, %for.inc46 ]
  br i1 undef, label %for.inc46, label %for.body5

for.body5:                                        ; preds = %for.cond2
  br i1 undef, label %lbl_465, label %if.then9

if.then9:                                         ; preds = %for.body5
  br label %return

lbl_465:                                          ; preds = %lbl_465, %for.body5
  %tobool19 = icmp eq i8 undef, 0
  br i1 %tobool19, label %if.end21, label %lbl_465

if.end21:                                         ; preds = %lbl_465
  %conv23 = zext i8 %p_44.addr.1 to i64
  %xor = xor i64 %conv23, 1
  %tobool.i = icmp eq i64 %conv23, 0
  br i1 %tobool.i, label %cond.false.i, label %safe_mod_func_uint64_t_u_u.exit

cond.false.i:                                     ; preds = %if.end21
  %div.i = udiv i64 %xor, %conv23
  br label %safe_mod_func_uint64_t_u_u.exit

safe_mod_func_uint64_t_u_u.exit:                  ; preds = %cond.false.i, %if.end21
  %cond.i = phi i64 [ %div.i, %cond.false.i ], [ %conv23, %if.end21 ]
  %tobool28 = icmp eq i64 %cond.i, 0
  br i1 %tobool28, label %lor.rhs, label %lor.end

lor.rhs:                                          ; preds = %safe_mod_func_uint64_t_u_u.exit
  br label %lor.end

lor.end:                                          ; preds = %lor.rhs, %safe_mod_func_uint64_t_u_u.exit
  br label %for.cond2

for.inc46:                                        ; preds = %for.cond2
  br label %for.cond2

if.end50:                                         ; preds = %entry
  br label %return

return:                                           ; preds = %if.end50, %if.then9
  ret void
}
