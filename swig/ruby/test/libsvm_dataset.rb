require 'test/unit'
require 'TinyClassifier'

# see below for datasets
# http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/

require 'open-uri'
require 'stringio'
require 'tempfile'
require 'uri'
def open_or_uri_open(target, &block)
  filename = Dir.tmpdir + File::SEPARATOR + URI.encode(target, /#{URI::UNSAFE}|\//).tr('%', '_')
  if !File.exists?(filename) then
    open(filename,'w') do |io|
      io.write open(target).read
    end
  end
  if block then
    open(filename) do |x|
      yield x
    end
  else
    return open(filename)
  end
end

if !Array.new.respond_to?(:shuffle) then
  def Array.shuffle
    a = self.clone
    a.shuffle!
    return a
  end
  def Array.shuffle!
    (0..self.length-1).each do |x|
      r = rand(self.length)
      self[x], self[r] = self[r], self[x]
    end
  end
end

class TC_LibSVM_Dataset < Test::Unit::TestCase
  include TinyClassifier

  DIM = 123
  DATA_TR = open_or_uri_open('http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/a1a').to_a.join
  DATA_TS = open_or_uri_open('http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/a1a.t').to_a.join

  def parse(io, &block)
    io.each_line do |line|
      line.gsub(/\#.*/,'')
      cols = line.split
      next if cols.length <= 0
      block.call(cols)
    end
  end

  def find_dim(io)
    dim = 0
    parse(io) do |cols|
      cols.shift
      cols.each do |d|
        ind,_ = *(d.split(/:/))
        ind = ind.to_i
        dim = ind + 1 if ind >= dim
      end
    end
    return dim
  end

  def libsvm_to_vec(io, dim=nil, &block)
    if !dim then
      str = io.read
      dim = find_dim(StringIO.new(str))
      io = StringIO.new(str)
    end
    labels  = IntVector.new
    vectors = FloatVectorVector.new
    parse(io) do |cols|
      vec = FloatVector.new(dim)
      lab = cols.shift.to_i
      cols.each do |d|
        ind,val = *(d.split(/:/))
        begin
          ind = ind.to_i
          next if ind >= dim
          vec[ind] = val.to_f
        rescue ArgumentError
          raise "'#{ind}:#{val}' at #{line} of #{io}"
        end
      end
      if block then
        yield lab, vec if block
      else
        vectors << vec
        labels  << lab
      end
    end
    if block then
      return nil
    else
      return [labels, vectors]
    end
  end

  def _test_libsvm(perp, train, test, dim)
    require 'stringio'

    train,test = *([train,test].map{|x| if x.is_a? String then StringIO.new(x)
                                        else x
                                        end})
    train = StringIO.new(train.to_a.shuffle.join)
    labels, vectors = *libsvm_to_vec(train, dim)
    STDERR.puts " training data size = #{labels.size}"
    if perp.respond_to?(:get_cache_size) then
      STDERR.puts " cache size = #{perp.get_cache_size}"
    end
    itr = perp.train(vectors, labels)
    STDERR.puts " iterations = #{itr}"

    pn = {
      true  => {:p => 0, :n => 0},
      false => {:p => 0, :n => 0}
    }
    libsvm_to_vec(test, dim) do |lab, vec|
      pred = perp.predict(vec)
      correct = lab * pred > 0
      #puts " ##{i+=1} #{correct}"
      pn[correct][lab > 0? :p : :n] += 1
    end
    STDERR.puts pn.inspect
    STDERR.puts " accuracy  = #{(pn[true][:p].to_f + pn[true][:n]) / (pn[true][:p] + pn[true][:n] + pn[false][:p] + pn[false][:n])}"
    prec = pn[true][:p].to_f / (pn[true][:p] + pn[false][:p])
    reca = pn[true][:p].to_f / (pn[true][:p] + pn[false][:n])
    STDERR.puts " precision = #{prec}"
    STDERR.puts " recall    = #{reca}"
    STDERR.puts " fmeasure  = #{1.0 / (0.5/prec + 0.5/reca)}"
  end

  def test_libsvm1
    require 'benchmark'
    Benchmark.bm do |bm|
      [['linear',    FloatPerceptron.new(DIM,   4)],
       ['1st-order', FloatPKProjectron.new(DIM, 4, 1, 0, 0, 0.8)],
       ['3rd-order', FloatPKPerceptron.new(DIM, 4, 3, 1, 0)]].each do |ent|
        name,p = *ent
        srand(1029)
        STDERR.puts "#{name}:"
        bm.report { _test_libsvm(p, DATA_TR, DATA_TS, DIM) }
      end
    end
  end
end
