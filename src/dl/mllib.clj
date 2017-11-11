(ns dl.mllib)

;------------------------------------------------------------------------------------------
(defn matrix [xs]
  "create a matrix from an array [1 2 3] => [[1] [2] [3]]"
  (partition 1 xs))
  ;(map #(list %) xs))
  ;(mapv #(vector %) xs))

(defn transpose [matrix]
  (apply map list matrix))
  ;(apply mapv vector matrix))

(defn product [a b]
  (doseq [x a] (assert (= (count x) (count b)) "The number of columns in first matrix should be equal to the number of rows in the second"))
  (doseq [x b] (assert (= (count (first b)) (count x)) "Number of columns in second matrix is not consistent"))
  (let [nested-for (fn [f x y] (map (fn [a] (map (fn [b] (f a b)) y)) x))]
    (nested-for (fn [x y] (reduce + (map * x y))) a (transpose b))))
  ;(let [nested-for (fn [f x y] (mapv (fn [a] (mapv (fn [b] (f a b)) y)) x))]
  ;  (nested-for (fn [x y] (reduce + (mapv * x y))) a (transpose b))))

(defn init-weight-matrix [x y]
  "create weight matrix inclusive n bias"
  (mapv #(repeatedly % rand) (repeat y (+ 1 x))))

;-----------------------------------------------------------------------------------------------------------------------
(defn feed-network
  ([network-input layer-weights activation-fn]
    ;(println "F=======================================================================================")
    ;(println "network-input    :" network-input)
    ;(println "layer-weights    :" layer-weights)
   (feed-network network-input layer-weights activation-fn (list (flatten network-input))))
  ([network-input layer-weights activation-fn layer-activations]
    ;(println "----------------------------------------------------------------------------------------")
    ;(println "network-input    :" network-input)
    ;(println "layer-activations:" layer-activations)
    ;(println "layer-weights    :" layer-weights)
   (if (empty? layer-weights)
     [network-input layer-activations] ; network-input => network-output!
     (let [activation (flatten (product (first layer-weights) (conj network-input [1.0])))
           output (map activation-fn activation)]
       (do
         ;(println "=>activation     :" activation)
         ;(println "=>output         :" output)
         (recur (matrix output) (rest layer-weights) activation-fn (conj layer-activations activation)))))))

(defn train-network [network-input layer-weights target rate activation-fn dactivation-fn]
  (let [[network-output layer-activations] (feed-network (matrix network-input) layer-weights activation-fn)
        prediction-error (map - (flatten target) (flatten network-output))
        inner-activations (rest layer-activations)
        layer-weights (reverse layer-weights)]
    (loop [activations inner-activations
           delta (matrix prediction-error)
           weights layer-weights
           result-gradients []]
      (if (empty? activations)
        [network-output (reverse
          (map (fn [a b] (map (fn [a b] (map (fn [a b] (+ a (* b rate))) a b)) a b))
               layer-weights result-gradients))]
        (let [[activation & next-activations] activations
              [weight & next-weights] weights
              activation-with-bias (conj (map activation-fn activation) 1.0)
              gradients (product delta [activation-with-bias])
              weight-without-bias (map (fn [xs] (rest xs)) weight)
              next-delta (map * (map dactivation-fn activation)
                              (flatten (product (transpose delta) weight-without-bias)))]
          (do
            ;(println "T=======================================================================================")
            ;(println "activation     :" activation)
            ;(println "weight         :" weight)
            ;(println "next-weights   :" next-weights)
            ;(println "delta          :" delta)
            ;(println "next-delta     :" next-delta)
            ;(println "next-activation:" next-activations)
            ;(println "gradients      :" gradients)
            (recur next-activations (matrix next-delta) next-weights (conj result-gradients gradients))))))))
