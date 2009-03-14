require 'mkmf'
SWIGNAME = 'tinyclassifier'
$INCFLAGS << ' -I../../include'
$LIBPATH << '../../lib'
if !have_library('stdc++') then
  mkmf_failed
end
if !have_library('tinyclassifier') then
  mkmf_failed
end

create_makefile(SWIGNAME)
