// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import Core library "io";
import Cpp library "<arpa/inet.h>";
import Cpp library "<netinet/in.h>";
import Cpp library "<stdio.h>";
import Cpp library "<string_view>";
import Cpp library "<sys/socket.h>";
import Cpp library "<unistd.h>";

import Cpp inline '''
#define SOCKADDR_IN_SIZE sizeof(struct sockaddr_in6)

// Work around htons being a macro, not a function, on MacOS.
// TODO: This should not be necessary once we can import function-like macros.
inline unsigned short htons_wrap(unsigned short n) {
  return htons(n);
}
''';

fn Perror(s: str) {
  // TODO: Should we allow passing a string literal to a `const char*`
  // parameter?
  Cpp.perror(Cpp.std.data(s));
}

// Basic example of using POSIX networking functions via interop.
//
// Opens a socket listening on port 8081, waits for a single connection, then
// reads a single chunk of up to 256 bytes from it and echoes it to the
// terminal.
fn Run() -> i32 {
  // TODO: `Cpp.SOCK_STREAM as i32` should probably be allowed.
  // TODO: Should this be an implicit conversion?
  let server_fd: i32 = Cpp.socket(Cpp.AF_INET6, (Cpp.SOCK_STREAM as u32) as i32, 0);
  if (server_fd == -1) {
    Perror("socket failed");
    return 1;
  }

  // TODO: Should be able to zero-initialize this with `= ()` or `= {}`.
  var address: Cpp.sockaddr_in6;
  Cpp.memset(&address, 0, Cpp.SOCKADDR_IN_SIZE);
  // TODO: Should this be valid without a cast? `AF_INET6` is defined to an
  // integer literal in glibc at least, but we import it as an `i32`.
  address.sin6_family = Cpp.AF_INET6 as Cpp.sa_family_t;
  // TODO: This leads to a link error referring to `in6addr_any.1`.
  // address.sin6_addr = Cpp.in6addr_any;
  let port: u16 = 8081;
  address.sin6_port = Cpp.htons_wrap(port);

  let address_ptr: Cpp.sockaddr* = &address unsafe as Cpp.sockaddr*;
  // TODO: Should have an implicit conversion from
  // T* to Optional(const T*).
  // TODO: Should expose `sizeof` somehow.
  let bind_result: i32 = Cpp.bind(
      server_fd, address_ptr as const Cpp.sockaddr*, Cpp.SOCKADDR_IN_SIZE as u32
  );

  if (bind_result == -1) {
      Perror("bind failed");
      return 1;
  }

  if (Cpp.listen(server_fd, 1) == -1) {
    Perror("listen failed");
    return 1;
  }

  // TODO: Add a better `Core.Print`.
  Core.PrintStr("Server is listening on port ");
  Core.Print(port as i32);

  var addr_len: Cpp.socklen_t = 0;
  let socket_fd: i32 = Cpp.accept(
      server_fd,
      address_ptr,
      &addr_len
  );

  if (socket_fd == -1) {
      Perror("accept failed");
      return 1;
  }

  Core.PrintStr("Connection accepted!\n");

  var buffer: array(char, 256);
  let len: Cpp.ssize_t = Cpp.read(socket_fd, &buffer[0], 256);
  if (len < 0) {
    Perror("read failed");
    return 1;
  }

  Cpp.write(1, &buffer[0], len as Cpp.size_t);

  Cpp.close(socket_fd);
  Cpp.close(server_fd);

  return 0;
}
