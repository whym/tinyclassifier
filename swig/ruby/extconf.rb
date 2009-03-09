require 'mkmf'
SWIGNAME = 'tinyclassifier'
$INCFLAGS << ' -I../../include'
$LIBPATH << '../../lib'
have_library('stdc++')
#have_library('tinyclassifier')
$DEFLIBPATH.delete('.')
create_makefile(SWIGNAME)
