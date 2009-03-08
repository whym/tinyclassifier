require 'test/unit'
require 'tinyclassifier'

class Numeric
  def positive?
    return self / abs > 0
  end
end
class TC_Perceptron < Test::Unit::TestCase
  include Tinyclassifier

  VECT = [1,3,4]
  MAT  = [[10,20,30],[40,50,60]]
  SAMPLES = {
    [-2, +1, -1] => 1,
    [-1, +2, +1] => 1,
    [-1, -1, -1] => 0,
    [+1, +1, -1] => 1,
    [+1, +2, -1] => 1
  }

  def test_poly_kernel
    x = IntVector.new(VECT)
    p = Perceptron.new(x.size)
    assert_equal((VECT.inject{|mem,y| mem + y*y}+1)**2,
                 p.kernel(x,x))
  end
  def test_vector
    assert_equal(MAT, IntVectorVector.new(MAT).map)
  end
  def test_power
    assert_equal(121, power_int(11,2))
    assert_equal((100*0.0121).to_i, (100*power_float(0.11,2)).to_i)
  end
  def test_train
    p = Perceptron.new(SAMPLES.keys[0].length)
    keys = SAMPLES.keys.sort
    p.train0(IntVectorVector.new(keys),
             BoolVector.new(keys.map{|x| SAMPLES[x]}))
    keys.each do |k|
      assert_equal(p.predict0(k).positive?, !SAMPLES[k].zero?)
    end
  end
end
