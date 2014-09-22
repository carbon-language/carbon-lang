// RUN: $(dirname %s)/check_clang_tidy_fix.sh %s google-readability-namespace-comments %t
// REQUIRES: shell

// CHECK-MESSAGES: :[[@LINE+2]]:11: warning: namespace not terminated with a closing comment [google-readability-namespace-comments]
// CHECK-MESSAGES: :[[@LINE+2]]:11: warning: namespace not terminated with a closing comment [google-readability-namespace-comments]
namespace n1 {
namespace n2 {

}
}
// CHECK-FIXES: }  // namespace n2
// CHECK-FIXES: }  // namespace n1


namespace short1 { namespace short2 { } }
