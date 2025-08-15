import bcrypt from "bcryptjs";
import jwt from "jsonwebtoken";
import User from "../models/User.js";
import WalletTransaction from "../models/WalletTransaction.js";
import generateTransactionId from "../utils/generateTransactionId.js";
import asyncHandler from "../utils/asyncHandler.js";

export const register = asyncHandler(async (req, res) => {
  const { name, phone, pin, confirmPin } = req.body;
  if (!name || !phone || !pin || !confirmPin) throw new Error("All fields required");
  if (!/^\d{10}$/.test(phone)) throw new Error("Invalid phone");
  if (!/^\d{4}$/.test(pin)) throw new Error("PIN must be 4 digits");
  if (pin !== confirmPin) throw new Error("PINs do not match");

  const exists = await User.findOne({ phone });
  if (exists) throw new Error("Phone already registered");

  const hashedPin = await bcrypt.hash(pin, 10);
  const user = await User.create({ name, phone, pin: hashedPin, balance: 1000 });

  await WalletTransaction.create({
    user: user._id,
    amount: 1000,
    type: "credit",
    description: "Welcome bonus",
    transactionId: generateTransactionId()
  });

  if (!process.env.JWT_SECRET) throw new Error("JWT_SECRET missing");
  const token = jwt.sign({ userId: user._id }, process.env.JWT_SECRET, { expiresIn: "24h" });

  res.status(201).json({ 
    token,
    user: { 
      id: user._id, 
      name, 
      phone, 
      balance: user.balance, 
      upiId: user.upiId 
    } 
  });
});

export const login = asyncHandler(async (req, res) => {
  const { phone, pin } = req.body;
  if (!phone || !pin) throw new Error("Phone and PIN required");
  const user = await User.findOne({ phone, isActive: true });
  if (!user) throw new Error("Invalid credentials");
  const ok = await bcrypt.compare(pin, user.pin);
  if (!ok) throw new Error("Invalid credentials");

  user.lastLogin = new Date();
  await user.save();

  if (!process.env.JWT_SECRET) throw new Error("JWT_SECRET missing");
  const token = jwt.sign({ userId: user._id }, process.env.JWT_SECRET, { expiresIn: "24h" });

  res.json({ 
    token, 
    user: { 
      id: user._id, 
      name: user.name, 
      phone: user.phone, 
      balance: user.balance, 
      upiId: user.upiId 
    } 
  });
});
