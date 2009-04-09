require 'test/unit'
require 'TinyClassifier'

# see below for datasets
# http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/

class TC_LibSVM_Dataset < Test::Unit::TestCase
  include TinyClassifier

  def libsvm_to_vec(io, dim, &block)
    labels  = IntVector.new
    vectors = FloatVectorVector.new
    io.each_line do |line|
      cols = line.split
      vec = FloatVector.new(dim)
      lab = cols.shift.to_i
      cols.each do |d|
        ind,val = *(d.split(/:/))
        begin
          vec[ind.to_i] = val.to_f
        rescue ArgumentError
          puts vec.inspect
          puts "'#{ind}:#{val}' at #{line}"
          exit 1
        end
      end
      vectors << vec
      labels  << lab
      yield lab, vec if block
    end
    return [labels, vectors]
  end

  def _test_libsvm(train, test, dim)
    require 'open-uri'
    labels, vectors = *libsvm_to_vec(open(train), dim)
    STDERR.puts "training data size = #{labels.size}"
    perp = FloatPerceptron.new(dim)
    perp.kernel_order = 5
    perp.train0(vectors, labels)

    pn = {
      true  => {:p => 0, :n => 0},
      false => {:p => 0, :n => 0}
    }
    i = 0
    libsvm_to_vec(open(test), dim) do |lab, vec|
      correct = lab * perp.predict0(FloatVector.new(vec)) > 0
      #puts "##{i+=1} #{correct}"
      pn[correct][lab > 0? :p : :n] += 1
    end
    puts pn.inspect
  end

  def test_libsvm1
    _test_libsvm('http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/a3a',
                 'http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/a3a.t',
                 123)
  end
end
