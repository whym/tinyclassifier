require 'test/unit'
require 'tinyclassifier'

class TC_Perceptron < Test::Unit::TestCase
  include Tinyclassifier

  VECT = [1,3,4]

  def test_poly_kernel
    x = IntVector.new(VECT)
    p = Perceptron.new(x.size)
    assert_equal(p.kernel(x,x),
                 VECT.inject{|mem,x| mem + x*x})
  end
end
