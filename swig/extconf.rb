require 'mkmf'
with_cppflags(['-I../include -L../lib -ltinyclassifier']) {
  with_ldflags(['-lstdc++ -ltinyclassifier']) {
    $DEFLIBPATH = $DEFLIBPATH.select{|x| x != '.'}
    create_makefile('tinyclassifier')
#     File.open('Makefile', 'a') do |io|
#       io.puts <<'EOD'
# tinyclassifier_wrap.cxx: tinyclassifier.i
# 	swig -I../include -Wall -c++ -ruby $<
# EOD
#    end
  }
}

# run this before "ruby extconf.rb":
#  swig -I../include -Wall -c++ -ruby $<
