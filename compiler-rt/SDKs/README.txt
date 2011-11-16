It is often convenient to be able to build compiler-rt libraries for a certain
platform without having a full SDK or development environment installed.

This makes it easy for users to build a compiler which can target a number of
different platforms, without having to actively maintain full development
environments for those platforms.

Since compiler-rt's libraries typically have minimal interaction with the
system, we achieve this by stubbing out the SDKs of certain platforms.
