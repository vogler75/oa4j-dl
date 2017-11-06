(ns dl.oa4j
  (:import (at.rocworks.oa4j.base JClient JManager IAnswer IHotLink JDpIdValueList JDpVCItem)
           (at.rocworks.oa4j.var DynVar Variable)))

; manager instance
(def manager (new JManager))

(defn startup [args]
  (.init manager (into-array args))
  (.start manager))

(defn convert-var [^Variable x]
  (if (instance? DynVar x)
    (map convert-var (.asList x))
    (.getValueObject x)))

(defn convert-vcitem [^JDpVCItem x]
    (convert-var (.getVariable x)))

(defn convert-dplist [^JDpIdValueList xs]
  (map convert-vcitem (.asList xs)))

(defn i-hotlink [f]
  (reify IHotLink
    (hotlink [this hotlink]
      (f (convert-dplist hotlink)))))

(defn i-answer [f]
  (reify IAnswer
    (answer [this answer]
      (f (convert-dplist answer)))))

(defn dpConnect
  ([dps callback]
    (dpConnect dps callback false))
  ([dps callback answer]
    (let [f (JClient/dpConnect)]
      (doall (map #(.add f (name %)) dps))
      (if answer (.answer f (i-answer callback)))
      (.hotlink f (i-hotlink callback))
      (.connect f))))

(defn dpSet
  ([kv] (let [f (JClient/dpSet)]
          ;(doall (map #(.add x (name (first %)) (last %)) kv))
          (doall (map (fn [[k v]] (.add f (name k) v)) kv))
          (.send f)))
  ([k v] (JClient/dpSet (name k) v)))

(defn dpGet [dps]
  (if (or (set? dps) (sequential? dps))
    (let [res (JClient/dpGet (map #(name %) dps))]
      (map #(convert-var %) res))
    (let [var (JClient/dpGet (name dps))]
      (convert-var var))))
