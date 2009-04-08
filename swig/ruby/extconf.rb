require 'mkmf'
SWIGNAME = 'TinyClassifier'
$INCFLAGS << ' -I../../include'
if !have_library('stdc++') then
  mkmf_failed
end
# $LIBPATH << '../../lib'
# if !have_library(SWIGNAME) then
#   mkmf_failed
# end

create_makefile(SWIGNAME)
