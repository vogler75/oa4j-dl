(ns dl.core
  (:gen-class)
  (:require [dl.oa4j :as scada]
            [dl.mllib :as ml]
            [clojure.data.json :as json])
  (:import (at.rocworks.oa4j.base JDpIdValueList JDpVCItem)
           (at.rocworks.oa4j.var DynVar)))

;;-----------------------------------------------------------------------------------------------------------------------
(defn activation-fn-tanh [x] "tanh activation" (Math/tanh x))
(defn dactivation-fn-tanh [y] "tanh derivative " (- 1.0 (* y y)))

(defn activation-fn-sigmoid [x] "sigmoid activation" (/ 1 (+ 1 (Math/exp (* -1 x)))))
(defn dactivation-fn-sigmoid [x] "sigmoid derivative" (/ (Math/exp x) (Math/pow (+ 1 (Math/exp x)) 2)))

;;-----------------------------------------------------------------------------------------------------------------------
(def nn-rate (atom 0.0))
(def nn-layers (atom (list)))
(def nn-weights (atom (list)))
(def nn-outputs (atom (list)))

(def nn-activation-fn (atom activation-fn-sigmoid))
(def nn-dactivation-fn (atom dactivation-fn-sigmoid))

;-----------------------------------------------------------------------------------------------------------------------
(defn scada-get-weights []
  (reset! nn-weights (json/read-str (scada/dpGet :Network.Network.Weights))))

(defn scada-set-weights []
  (scada/dpSet :Network.Network.Weights (json/write-str @nn-weights)))

(defn scada-set-output []
  (scada/dpSet :Network.Data.Output (DynVar. (to-array (first @nn-outputs)))))

;-----------------------------------------------------------------------------------------------------------------------
; layers [2 3 2]

; weights layer 1 (input to hidden 2=>3)
;[[0.12 0.01]
; [0.20 0.02]
; [0.13 0.03]]

; weights layer 2 (hidden to output 3=>2)
;[[0.15 0.02 0.01]
; [0.16 0.03 0.02]]

(defn init-weight-matrix [x y]
  "create weight matrix inclusive n bias"
  (mapv #(repeatedly % rand) (repeat y (+ 1 x))))

(defn init-network [layers]
  (println "init-network" layers)
  (reset! nn-layers layers)
  (reset! nn-weights (mapv init-weight-matrix layers (rest layers))))


;-----------------------------------------------------------------------------------------------------------------------
(defn feed-network
  ([network-input layer-weights activation-fn]
    (println "F=======================================================================================")
    (println "network-input    :" network-input)
    (println "layer-weights    :" layer-weights)
    (feed-network network-input layer-weights activation-fn (vector (flatten network-input))))
  ([network-input layer-weights activation-fn layer-activations]
    (println "----------------------------------------------------------------------------------------")
    (println "network-input    :" network-input)
    (println "layer-activations:" layer-activations)
    (println "layer-weights    :" layer-weights)
    (if (empty? layer-weights)
      [network-input layer-activations]
      (let [activation (flatten (ml/product (first layer-weights) (conj network-input [1.0])))
            output (mapv activation-fn activation)]
        (do
          (println "=>activation     :" activation)
          (println "=>output         :" output)
          (recur (ml/matrix output) (rest layer-weights) activation-fn (conj layer-activations activation)))))))

;-----------------------------------------------------------------------------------------------------------------------
(defn train-backwards [output-error activations weights activation-fn activation-fn-derivative output-weights]
  (println "TB=======================================================================================")
  (println "output-error     :" output-error)
  (println "activations      :" activations)
  (println "weights          :" weights)
  (println "output-weights   :" output-weights)
  (if (empty? activations)
    output-weights
    (let [activation (conj (vec (first activations)) 1.0)
          weight (first weights)
          hidden-error (ml/hidden-errors-calculation activation output-error weight @nn-dactivation-fn)
          hidden-weight (ml/weights-calculation activation hidden-error weight @nn-rate)
          rest-outputs (rest activations)
          rest-weights (rest weights)]
      (do
        (recur hidden-error rest-outputs rest-weights activation-fn activation-fn-derivative (conj output-weights hidden-weight))))))


(defn train-network [network-input layer-weights target activation-fn dactivation-fn]
  (let [[network-output layer-activations] (feed-network (ml/matrix network-input) layer-weights activation-fn)
        prediction-error (mapv - (flatten target) (flatten network-output))
        inner-activations (rest (reverse layer-activations))
        layer-weights (reverse layer-weights)]
    (loop [activations inner-activations
           delta (ml/matrix prediction-error)
           weights layer-weights
           result-gradients []]
      (if (empty? activations)
        result-gradients
        (let [[activation & next-activations] activations
              [weight & next-weights] weights
              activation-with-bias (conj (vec activation) 1.0)
              gradients (ml/product delta [activation-with-bias])
              next-delta (mapv *
                               (mapv dactivation-fn activation)
                               (flatten (ml/product (ml/transpose delta) weight)))]
          (do
            (println "----------------------------------------------------------------------------------------")
            (println "activation     :" activation)
            (println "weight         :" weight)
            (println "next-weights   :" next-weights)
            (println "delta          :" delta)
            (println "next-delta     :" next-delta)
            (println "next-activation:" next-activations)
            (println "gradients      :" gradients)
            (recur next-activations (ml/matrix next-delta) next-weights (conj result-gradients gradients))))))))

;-----------------------------------------------------------------------------------------------------------------------
(defn reset-callback [[layers]]
  (init-network layers))

(defn set-activation-fn-callback [[nr]]
  (case nr
    0 (do
        (reset! nn-activation-fn activation-fn-tanh)
        (reset! nn-dactivation-fn dactivation-fn-tanh))
    1 (do
        (reset! nn-activation-fn activation-fn-sigmoid)
        (reset! nn-dactivation-fn dactivation-fn-sigmoid))))

(defn set-rate-callback [[rate]]
  (reset! nn-rate rate))

(defn save-network-callback [[trigger]]
  (scada-set-weights))

(defn feed-callback [[input target train]]
  (if train
    (reset! nn-weights (train-network input @nn-weights target @nn-activation-fn @nn-dactivation-fn))
    (reset! nn-outputs (feed-network (ml/matrix input) @nn-weights @nn-activation-fn)))
  (scada-set-output))

;-----------------------------------------------------------------------------------------------------------------------
(def xor-data [{:i [0 0] :t [0]}
               {:i [0 1] :t [1]}
               {:i [1 0] :t [1]}
               {:i [1 1] :t [0]}])

(defn xor-test []
  (doall
    (map
      #(println (:i %) "=>" (feed-network (ml/matrix (:i %)) @nn-weights @nn-activation-fn))
                xor-data)))

;-----------------------------------------------------------------------------------------------------------------------
(defn -main [& args]
  (comment
    (scada/startup args)

    (scada-get-weights)
    (println "weights" @nn-weights)

    (scada/dpConnect [:Network.Config.Layers] reset-callback)
    (scada/dpConnect [:Network.Config.ActivationFn] set-activation-fn-callback true)

    (scada/dpConnect [:Network.Control.Rate] set-rate-callback true)

    (scada/dpConnect [:Network.Data.Input
                      :Network.Data.Target
                      :Network.Control.Train] feed-callback)

    (scada/dpConnect [:Network.Control.Save] save-network-callback)
    )


  ; already trained network for XOR
  (reset! nn-weights [[[  -3.9050378273319755  -3.8927622364550345 5.986938319255103]
                       [-5.049468802938935   -5.0079142506851575 1.99893455091257   ]]
                      [[14.718764645089621  -14.907682119185624 -7.04504038559793  ]]])

  ; xor-test
  ;(init-network [2 2 1])

  ;(dotimes [_ 10000]
  ;  (doseq [x xor-data]
  ;    (reset! nn-weights (train-network (:i x) @nn-weights (:t x) @nn-activation-fn @nn-dactivation-fn))))

  (let [x (first xor-data)]
    (println "gradients: " (train-network (:i x) @nn-weights (:t x) @nn-activation-fn @nn-dactivation-fn)))

  ;@nn-weights

  ;(xor-test)

  (println "ready")
)


