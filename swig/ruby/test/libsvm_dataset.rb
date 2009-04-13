require 'test/unit'
require 'TinyClassifier'

# see below for datasets
# http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/

require 'open-uri'
require 'stringio'
class TC_LibSVM_Dataset < Test::Unit::TestCase
  include TinyClassifier

  DIM = 123
  DATA_TR = open('http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/a1a').to_a.join
  DATA_TS = open('http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/a1a.t').to_a.join

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

    train,test = *([train,test].map{|x| if x.is_a? String then StringIO.new(x)
                                        else x
                                        end})
    train = StringIO.new(train.to_a.shuffle.join)
    labels, vectors = *libsvm_to_vec(train, dim)
    STDERR.puts "training data size = #{labels.size}"
    itr = perp.train(vectors, labels)
    STDERR.puts "iterations = #{itr}"

    pn = {
      true  => {:p => 0, :n => 0},
      false => {:p => 0, :n => 0}
    }
    libsvm_to_vec(test, dim) do |lab, vec|
      pred = perp.predict(vec)
      correct = lab * pred > 0
      #puts "##{i+=1} #{correct}"
      pn[correct][lab > 0? :p : :n] += 1
    end
    STDERR.puts pn.inspect
    STDERR.puts "accuracy   = #{(pn[true][:p].to_f + pn[true][:n]) / (pn[true][:p] + pn[true][:n] + pn[false][:p] + pn[false][:n])}"
    prec = pn[true][:p].to_f / (pn[true][:p] + pn[false][:p])
    reca = pn[true][:p].to_f / (pn[true][:p] + pn[false][:n])
    STDERR.puts "precision = #{prec}"
    STDERR.puts "recall    = #{reca}"
    STDERR.puts "fmeasure  = #{1.0 / (0.5/prec + 0.5/reca)}"
  end

  def test_libsvm1
    [FloatPerceptron.new(DIM, 4),
     FloatPKPerceptron.new(DIM, 4, 1, 0, 1000*1000*100),
     FloatPKPerceptron.new(DIM, 4, 5, 1)].each do |p|
      srand(1029)
      _test_libsvm(p, DATA_TR, DATA_TS, DIM)
    end
  end
end
