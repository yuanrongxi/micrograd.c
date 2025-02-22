local math = require("math")

local RNG = {}

function RNG.new(seed)
    local self ={seed = seed}
    setmetatable(self, { __index = RNG })
    return self
end

function RNG:random_u32()
    self.seed = self.seed ~ (self.seed >> 12) & 0xFFFFFFFFFFFFFFFF
    self.seed = self.seed ~ (self.seed << 25) & 0xFFFFFFFFFFFFFFFF
    self.seed = self.seed ~ (self.seed >> 27) & 0xFFFFFFFFFFFFFFFF
    return (self.seed * 0x2545F4914F6CDD1D >> 32) & 0xFFFFFFFF
end

function RNG:rand()
    return (self:random_u32() >> 8) / 16777216.0
end

function RNG:uniform(a, b)
    return a + (b - a) * self:rand()
end

local rng = RNG.new(42)

function dist_to_right_dot(x, y, r_big)
    return math.sqrt((x - 1.5 * r_big) ^ 2 + (y - r_big) ^ 2)
end

function dist_to_left_dot(x, y, r_big)
    return math.sqrt((x - 0.5 * r_big) ^ 2 + (y - r_big) ^ 2)
end

function which_class(x, y, r_small, r_big)
    local d_right = dist_to_right_dot(x, y, r_big)
    local d_left = dist_to_left_dot(x, y, r_big)
    local criterion1 = d_right <= r_small
    local criterion2 = d_left > r_small and d_left <= 0.5 * r_big
    local criterion3 = y > r_big and d_right > 0.5 * r_big
    local is_yin = criterion1 or criterion2 or criterion3
    local is_circles = d_right < r_small or d_left < r_small

    if is_circles then
        return 2
    end

    if is_yin then
        return 0
    else
        return 1
    end
end

function get_sample(goal_class, r_small, r_big)
    while true do
        local x = rng:uniform(0, 2 * r_big)
        local y = rng:uniform(0, 2 * r_big)
        if math.sqrt((x - r_big) ^ 2 + (y - r_big) ^ 2) <= r_big then
            local c = which_class(x, y, r_small, r_big)
            if goal_class == nil or c == goal_class then
                local scaled_x = (x / r_big - 1) * 2
                local scaled_y = (y / r_big - 1) * 2
                return scaled_x, scaled_y, c
            end
        end
    end
end


function gen_data_yinyang(n, r_small, r_big)
    local pts = {}
    for i = 1, n do
        local goal_class = i % 3
        local x, y, c = get_sample(goal_class, r_small, r_big)
        pts[i] = {{x, y}, c}
    end
    return pts
end



local value = {}

function value.new(data, _prev, _op)
    local self = setmetatable({}, value)
    self.data = data
    self.grad = 0
    self.m = 0
    self.v = 0
    self._prev = _prev or {}
    self._op = _op or ''
    self._backward = function() end
    
    return self
end

function value:add(b)
    b = type(b) == "number" and value.new(b) or b
    local out = value.new(self.data + b.data, {self, b}, "+")
    out._backward = function()
        self.grad = self.grad + out.grad
        b.grad = b.grad + out.grad
    end
    return out
end 

function value:mul(b)
    b = type(b) == "number" and value.new(b) or b
    local out = value.new(self.data * b.data, {self, b}, "*")
    out._backward = function()
        self.grad = self.grad + out.grad * b.data
        b.grad = b.grad + out.grad * self.data
    end
    return out
end

function value:neg()
    local a = value.new(-1)
    return self:mul(a)
end


function value:sub(b)
    b = type(b) == "number" and value.new(b) or b
    local out = value.new(self.data - b.data, {self, b}, "-")
    out._backward = function()
        self.grad = self.grad + out.grad
        b.grad = b.grad - out.grad
    end
    return out
end

function value:div(b)
    b = type(b) == "number" and value.new(b) or b
    local out = value.new(self.data / b.data, {self, b}, "/")
    out._backward = function()  
        self.grad = self.grad + out.grad / b.data
        b.grad = b.grad - out.grad * self.data / b.data ^ 2 
    end
    return out
end

function value_pow(val, b)
    b = type(b) == "number" and value.new(b) or b
    local out = value.new(val.data ^ b.data, {val, b}, "**")
    out._backward = function()
        val.grad = val.grad + out.grad * b.data * val.data ^ (b.data - 1)
    end
    return out
end



function value_relu(val)
    local out = value.new(val.data > 0 and val.data or 0, {val}, "ReLU")
    out._backward = function()
        val.grad = val.grad + out.grad * (val.data > 0 and 1 or 0)
    end
    return out
end

function value_tanh(val)
    local out = value.new(math.tanh(val.data), {val}, "tanh")
    out._backward = function()
        val.grad = val.grad + out.grad * (1 - out.data ^ 2)
    end
    return out
end

function value_exp(val)
    local out = value.new(math.exp(val.data), {val}, "exp")
    out._backward = function()
        val.grad = val.grad + out.grad * math.exp(val.data)
    end
    return out
end

function value_log(val)
    local out = value.new(math.log(val.data), {val}, "log")
    out._backward = function()
        val.grad = val.grad + out.grad * (1 / val.data)
    end
    return out
end

function value_backward(val)
    local topo = {}
    local visited = {}
    
    local function build_topo(v)
        if visited[v] then return end
        visited[v] = true
        for _, child in pairs(v._prev) do
            build_topo(child)
        end
        table.insert(topo, v)   
    end
    
    build_topo(val)
    
    val.grad = 1.0
    
    for i = #topo, 1, -1 do 
        topo[i]._backward()
    end
end

value.__add = value.add
value.__mul = value.mul
value.__sub = value.sub
value.__div = value.div
value.__unm = value.neg

local adamw = {}

function adamw.new(lr, weight_decay)
    local self = {lr = lr, beta1 = 0.9, beta2 = 0.95, eps = 1e-8, weight_decay = weight_decay, t = 0}
    setmetatable(self, { __index = adamw })
    return self
end

function adamw:step(val)
    local m = self.beta1 * val.m + (1 - self.beta1) * val.grad
    local v = self.beta2 * val.v + (1 - self.beta2) * val.grad ^ 2
    local m_hat = m / (1 - self.beta1 ^ self.t)
    local v_hat = v / (1 - self.beta2 ^ self.t)
    local data = val.data - self.lr * (m_hat / (v_hat ^ 0.5 + 1e-8) + self.weight_decay * val.data)

    return m, v, data
end



local Neuron = {}

function Neuron.new(nin, nonlin)
    local self = {}
    setmetatable(self, { __index = Neuron })
    self.w = {}
    for i = 1, nin do
        self.w[i] = value.new(rng:uniform(-1, 1) * nin ^ -0.5)
    end
    self.b = value.new(0)
    self.nonlin = nonlin
    return self
end

function Neuron:forward(x)
    local out = value.new(0)
    for i = 1, #self.w do
        out = out + self.w[i] * x[i]
    end
    
    out = out + self.b
    
    if self.nonlin then
        return value_tanh(out)
    else
        return out
    end
end

function Neuron:step(ad)
    for i = 1, #self.w do   
        self.w[i].m, self.w[i].v, self.w[i].data = ad:step(self.w[i])
    end
    self.b.m, self.b.v, self.b.data = ad:step(self.b)
end 

function Neuron:zero_grad()
    for i = 1, #self.w do
        self.w[i].grad = 0
    end
    self.b.grad = 0
end
local Layer = {}

function Layer.new(nin, nout, nonlin)
    local self = {}
    setmetatable(self, { __index = Layer })

    self.neurons = {}
    for i = 1, nout do
        self.neurons[i] = Neuron.new(nin, nonlin)
    end
    return self
end

function Layer:forward(x)
    local out = {}
    for i = 1, #self.neurons do
        out[i] = self.neurons[i]:forward(x)
    end
    return out
end

function Layer:step(ad)
    for i = 1, #self.neurons do
        self.neurons[i]:step(ad)
    end
end

function Layer:zero_grad()
    for i = 1, #self.neurons do
        self.neurons[i]:zero_grad()
    end
end

local MLP = {}

function MLP.new(nin, nouts)
    local self = {}
    setmetatable(self, { __index = MLP })
    
    self.layers = {}
    local sz = {nin, table.unpack(nouts)}
    for i = 1, #sz - 1 do
        self.layers[i] = Layer.new(sz[i], sz[i+1], i~=#nouts)
    end
    return self
end

function MLP:forward(x)
    for i = 1, #self.layers do
        x = self.layers[i]:forward(x)
    end
    return x
end

function MLP:step(ad)
    for i = 1, #self.layers do
        self.layers[i]:step(ad)
    end
end

function MLP:zero_grad()
    for i = 1, #self.layers do
        self.layers[i]:zero_grad()
    end
end


function cross_entropy(logits, target)
    local ex = {}
    for i = 1, #logits do
        ex[i] = value_exp(logits[i])
    end

    local denom = value.new(0)
    for i = 1, #ex do
        denom = denom + ex[i]
    end

    local probs = {}
    for i = 1, #ex do
        probs[i] = ex[i] / denom
    end

    local logp = value_log(probs[target + 1])
    local t = value.new(0)
    local nll = t - logp
    return nll
end

-- model and optimizer
local count  = 1000
local train_split = gen_data_yinyang(count * 8 / 10, 0.1, 0.5)
local val_split = gen_data_yinyang(count * 1 / 10, 0.1, 0.5)
local test_split = gen_data_yinyang(count * 1 / 10, 0.1, 0.5)

local model = MLP.new(2, {8, 3})
local optimizer = adamw.new(1e-1, 1e-4)

function loss_fun(split)
    local total_loss = value.new(0)
    for i = 1, #split do
        local logits = model:forward(split[i][1])
        local loss = cross_entropy(logits, split[i][2])
        total_loss = total_loss + loss
    end
    --local f = value.new(1.0 / #split)
    --local mean_loss = total_loss * f
    local mean_loss = total_loss/#split
    return mean_loss
end


function train(ad)
    local num_steps = 100

    local start_time = os.time()
    for i = 1, num_steps do
        if i % 10 == 0 then
            local val_loss = loss_fun(val_split)
            print(string.format("step %d/%d, val loss:%.6f", i, num_steps, val_loss.data))
        end
        local loss = loss_fun(train_split)
        value_backward(loss)
        
        optimizer.t = optimizer.t + 1
        model:step(optimizer)
        model:zero_grad()

        print(string.format("step %d/%d, train loss:%.6f", i, num_steps, loss.data))
    end

    local end_time = os.time()
    print(string.format("Time taken(steps:%d, samples:%d): %.6f seconds", num_steps, count, end_time - start_time))
end


function test()
    local a = value.new(-4.0)
    local b = value.new(2) 

    local c = a + b
    local d = a * b + value_pow(b, 3)
    c = c + 1
    c = c + 1 - a
    d = d * 2 + value_relu(b + a)
    d = d * 3 + value_relu(b - a)
    local e = c - d
    local f = e * 2
    local g = value_exp(f) 
    local h = value.new(10.0)
    g = g + h / f
    value_backward(g)

    print(string.format("g.data:%.6f", g.data))

   print(string.format("a.grad:%.6f", a.grad))
   print(string.format("b.grad:%.6f", b.grad))

    for i = 1, #val_split do
        print(string.format("x:%.6f, y:%.6f, c:%d", val_split[i][1][1], val_split[i][1][2], val_split[i][2]))
    end

end

--test()

train(optimizer)





