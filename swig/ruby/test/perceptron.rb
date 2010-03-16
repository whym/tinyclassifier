require 'test/unit'
require 'TinyClassifier'
require 'tempfile'

class Numeric
  def polarity
    self > 0 ? +1: -1
  end
end
class TC_Perceptron < Test::Unit::TestCase
  include TinyClassifier

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

  def setup
    @samples = SAMPLES.keys
    @labels = SAMPLES.keys.map{|x| SAMPLES[x]}
  end

  def setup_samples
    dim = 100
    pivs = [0,1,8,9,15]
    @samples = IntVectorVector.new
    sample_num = 100
    @labels = IntVector.new(sample_num)
    sample_num.times do |i|
      v = (1..dim).map{ ((rand - 0.5)*6).to_i }
      @samples << IntVector.new(v)
      @labels[i] = (pivs.inject{|mem,j| mem and (v[j]/2) > 0})? +1 : -1
    end
  end

  def test_poly_kernel
    x = IntVector.new(VECT)
    p = IntPKPerceptron.new(x.size)
    assert_equal((VECT.inject{|mem,y| mem + y*y}+1)**2,
                 p.kernel(x,x))
  end
  def test_vector
    assert_equal(MAT, IntVectorVector.new(MAT).map)
  end
  def test_power
    assert_equal(11**2, power_int(11,2))
    a = (0..4).to_a
    assert_equal(a.map{|x| 2**x}, a.map{|x| power_int(2,x)})
    assert_equal((100*(0.11**2)).to_i, (100*power_float(0.11,2)).to_i)
  end

  def test_train
    [IntPerceptron.new(SAMPLES.keys[0].length, 10),
     IntPKPerceptron.new(SAMPLES.keys[0].length, 10)].each do |p|
      keys = SAMPLES.keys.sort
      p.train(IntVectorVector.new(keys),
              IntVector.new(keys.map{|x| SAMPLES[x]}))
      assert_equal(keys.map{|k| [k, SAMPLES[k]]},
                   keys.map do |k|
                   pred = p.predict(k)
                     [k, pred.polarity]
                   end)
    end
  end
  def test_train_big
    setup_samples
    [IntPerceptron.new(@samples[0].length, 100),
     IntPKPerceptron.new(@samples[0].length, 100)].each do |p|
      p.train(@samples, @labels)
      assert_equal((0..@samples.size-1).map{|i| [@samples[i], @labels[i]]},
                   (0..@samples.size-1).map{|i| [@samples[i], p.predict(@samples[i]).polarity]})
    end
  end

  def test_save
    #setup_samples
    p1 = IntPerceptron.new(@samples[0].length, 100)
    p1.train(@samples, @labels)
    f = 'xx'#Tempfile.new('model').path
    p1.store(f)
    p2 = IntPerceptron.new(@samples[0].length, 100)
    p2.load(f)
    assert_equal((0..@samples.size-1).map{|i| [@samples[i], p1.predict(@samples[i]).polarity]},
                 (0..@samples.size-1).map{|i| [@samples[i], p2.predict(@samples[i]).polarity]})
  end
end
