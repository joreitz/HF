use std::env;
use std::io::{self, BufRead};

pub mod io_tools {
    use super::*;

    /// Reads a line from any type that implements `BufRead` (like a File or Stdin) into `line`.
    /// 
    /// This is the idiomatic Rust equivalent of the Fortran `read_line` subroutine.
    /// It dynamically allocates the string as needed, completely avoiding the need 
    /// for manual 128-byte chunking.
    /// 
    /// Returns an `io::Result<usize>` representing the number of bytes read.
    /// An Ok(0) indicates End of File (EOF).
    pub fn read_line<R: BufRead>(reader: &mut R, line: &mut String) -> io::Result<usize> {
        // Clear the buffer before reading, matching the `line = ''` Fortran behavior
        line.clear();
        
        // read_line safely handles reading until a newline, including dynamic allocation
        reader.read_line(line)
    }

    /// Reads the `iarg`'th command-line argument.
    /// 
    /// In Rust, `std::env::args()` returns an iterator where the 0th element
    /// is the program executable name, and subsequent elements are the arguments.
    /// This aligns perfectly with Fortran's `get_command_argument` indexing.
    /// 
    /// Returns `Some(String)` if the argument is found, or `None` if it does not exist.
    pub fn read_argument(iarg: usize) -> Option<String> {
        env::args().nth(iarg)
    }
}