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

  def _test_libsvm(perp, train, test, dim)
    require 'open-uri'
    require 'stringio'
    labels, vectors = *libsvm_to_vec(StringIO.new(open(train).to_a.shuffle.join), dim)
    STDERR.puts "training data size = #{labels.size}"
    itr = perp.train(vectors, labels)
    STDERR.puts "iterations #{itr}"

    pn = {
      true  => {:p => 0, :n => 0},
      false => {:p => 0, :n => 0}
    }
    i = 0
    libsvm_to_vec(open(test), dim) do |lab, vec|
      correct = lab * perp.predict(FloatVector.new(vec)) > 0
      #puts "##{i+=1} #{correct}"
      pn[correct][lab > 0? :p : :n] += 1
    end
    puts pn.inspect
  end

  def test_libsvm
    dim, data_tr, data_ts = *[123,
                              'http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/a1a',
                              'http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/a1a.t']
    srand(1029)
    [FloatPerceptron.new(dim, 5),
     FloatPKPerceptron.new(dim, 5, 3),
     FloatPKPerceptron.new(dim, 5, 5)].each do |p|
      _test_libsvm(p, data_tr, data_ts, dim)
    end
  end
end
