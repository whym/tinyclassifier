require 'test/unit'
require 'tinyclassifier'

class Numeric
  def positive?
    (self.abs > 0) and ((self / self.abs) > 0)
  end
end
class TC_Perceptron < Test::Unit::TestCase
  include Tinyclassifier

  VECT = [1,3,4]
  MAT  = [[10,20,30],[40,50,60]]
  SAMPLES = {
    [-2, +1, -1] => +1,
    [-1, +2, +1] => +1,
    [-1, -1, -1] => -1,
    [+1, +1, -1] => +1,
    [-1, +1, -1] => +1,
    [+1, -2, -1] => -1,
    [+1, -1, +1] => -1
  }

  def test_poly_kernel
    x = IntVector.new(VECT)
    p = IntPerceptron.new(x.size)
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
    p = IntPerceptron.new(SAMPLES.keys[0].length)
    keys = SAMPLES.keys.sort
    p.train0(IntVectorVector.new(keys),
             IntVector.new(keys.map{|x| SAMPLES[x]}))
    assert_equal(keys.map{|k| [k, SAMPLES[k]]},
                 keys.map do |k|
                   pred = p.predict0(k)
                   [k, pred.positive?() ?+1: -1]
                 end)
  end
  def test_train_big
    dim = 100
    pivs = [0,1,8,9,15]
    p = IntPerceptron.new(dim, 1000)
    samples = IntVectorVector.new
    sample_num = 100
    labels = IntVector.new(sample_num)
    sample_num.times do |i|
      v = (1..dim).map{ ((rand - 0.5)*6).to_i }
      samples << IntVector.new(v)
      labels[i] = (pivs.inject{|mem,j| mem and (v[j]/2).positive?})? +1 : -1
    end
    p.train0(samples, labels)
    assert_equal((0..sample_num-1).map{|i| [samples[i], labels[i]]},
                 (0..sample_num-1).map{|i| [samples[i], p.predict0(samples[i]).positive?() ?+1:-1]})
  end
end
