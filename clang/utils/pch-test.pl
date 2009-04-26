#!/usr/bin/perl -w

# This tiny little script, which should be run from the clang
# directory (with clang-cc in your patch), tries to take each
# compilable Clang test and build a PCH file from that test, then read
# and dump the contents of the PCH file just created.
use POSIX;

$exitcode = 0;

sub testfiles($$) {
  my $suffix = shift;
  my $language = shift;

  @files = `ls test/*/*.$suffix`;
  foreach $file (@files) {
    chomp($file);
    print(".");
    my $code = system("clang-cc -fsyntax-only -x $language $file > /dev/null 2>&1");
    if ($code == 0) {
      $code = system("clang-cc -emit-pch -x $language -o $file.pch $file > /dev/null 2>&1");
      if ($code == 0) {
        $code = system("clang-cc -include-pch $file.pch -x $language -ast-dump-full /dev/null > /dev/null 2>&1");
        if ($code == 0) {
        } elsif (($code & 0xFF) == SIGINT) {
          exit($exitcode);
        } else {
          print("\n---Failed to dump AST file for \"$file\"---\n");
          $exitcode = 1;
        }
        unlink "$file.pch";
      } elsif (($code & 0xFF) == SIGINT) {
        exit($exitcode);
      } else {
        print("\n---Failed to build PCH file for \"$file\"---\n");
        $exitcode = 1;
      }
    } elsif (($code & 0xFF) == SIGINT) {
      exit($exitcode);
    }
  }
}

printf("-----Testing precompiled headers for C-----\n");
testfiles("c", "c");
printf("\n-----Testing precompiled headers for Objective-C-----\n");
testfiles("m", "objective-c");
print("\n");
exit($exitcode);
